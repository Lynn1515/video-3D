import torch
import os
from glob import glob
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
#model = VGGT.from_pretrained("/home/cx_wchn/lnwang/ViewCrafter/checkpoints/vggt_model.pt").to(device)
# 1. 初始化模型结构
model = VGGT().to(device)

# 2. 直接加载本地权重
checkpoint_path = "/home/cx_wchn/lnwang/ViewCrafter/checkpoints/vggt_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)


# Load and preprocess example images (replace with your own image paths)
# 指定图像文件夹路径
image_dir = "/home/cx_wchn/lnwang/ViewCrafter/output_0505/extracted"

# image_names = ["/home/cx_wchn/lnwang/ViewCrafter/test/images_sparse/aleks-teapot-sparse/0.png", 
#                "/home/cx_wchn/lnwang/ViewCrafter/test/images_sparse/aleks-teapot-sparse/1.png"]  

# 读取文件夹下所有图片（支持常见格式：png, jpg, jpeg）
image_names = sorted(
    glob(os.path.join(image_dir, "*.[pjPj][pnPN]*"))
)

print(f"Found {len(image_names)} images:")

images = load_and_preprocess_images(image_names).to(device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()  # 切换到 eval 模式

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        # print(predictions)  #pose_enc(1,2,9), pose_enc_list长度为4list, depth（1，2，518，518，1）,depth_conf（1，2，518，518）, 
        # #world points(1,2,518,518,3) ,world points_conf(1,2,518,518), images(1,2,3,518,518)

        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)

         # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    #extrinsic (batch, num_images,3,4)  3x4矩阵 intrinsic (batch, num_images,3,3)  3x3矩阵

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    #depth（batch, num_images,518，518，1）,depth_conf（batch, num_images, 518，518）,
    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    #(batch, num_images,518,518,3) ,points_conf(batch, num_images,518,518)
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                              intrinsic.squeeze(0))#(num_images,518,518,3)      
    print('lx great!')
        