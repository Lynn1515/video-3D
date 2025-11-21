import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
import trimesh
import pandas as pd

device = "cuda:0"

# -------------------------------
# 渲染相关函数
# -------------------------------
def setup_renderer(cameras, image_size):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,
        points_per_pixel=10,
        bin_size=0
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    return renderer

def render_pcd(pts3d, colors, cameras, renderer, device):
    pts = torch.tensor(pts3d, dtype=torch.float32, device=device)
    col = torch.tensor(colors, dtype=torch.float32, device=device)
    point_cloud = Pointclouds(points=[pts], features=[col])
    images = renderer(point_cloud)
    return images

# -------------------------------
# Step 0: 读取点云和彩色信息
# -------------------------------
ply_file = "/home/cx_wchn/lnwang/ViewCrafter/output/20250708_2040_aleks-teapot/pcd0.ply"
mesh = trimesh.load(ply_file)

points_np = np.asarray(mesh.vertices)  # [N,3]
colors_np = mesh.visual.vertex_colors[:, :3] / 255.0 if mesh.visual.vertex_colors is not None else np.ones_like(points_np)

# 参考图像
ref_img = plt.imread("test/images/aleks-teapot.png")
ref_img = resize(ref_img, (288, 512), anti_aliasing=True)  # resize
image_h, image_w = ref_img.shape[:2]
#960,536,3

focal_length = 542.4215
center = np.array([256, 144])

# 参考视角世界到相机矩阵
c2w = np.array([[ 1.0000e+00,  4.1630e-07,  1.4324e-06, -1.2442e-08],
                [-5.3952e-07, -9.9609e-01, -8.7158e-02,  2.2842e-02],
                [-1.3905e-06,  8.7158e-02, -9.9609e-01,  2.6099e-01],
                [ 0.0, 0.0, 0.0, 1.0]])
ref_dir = -c2w[:3,2]

# -------------------------------
# Step 1: 难度计算函数
# -------------------------------
def compute_view_diff(points_np, K, R, t, ref_dir, image_w, image_h, alpha=0.6, beta=0.3, gamma=0.1):
    P_cam = (R @ points_np.T + t).T
    z = P_cam[:, 2]
    uv_h = (K @ P_cam.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < image_w) & (uv[:, 1] >= 0) & (uv[:, 1] < image_h)
    vis_ratio = mask.sum() / len(points_np)
    depth_var = np.var(z[mask]) if mask.sum() > 0 else 1.0
    cam_dir = -R[:, 2]
    angle = np.arccos(np.clip(np.dot(ref_dir, cam_dir), -1.0, 1.0))
    angle_norm = angle / np.pi
    return alpha * (1 - vis_ratio) + beta * depth_var + gamma * angle_norm

# -------------------------------
# Step 2: 构造候选视角
# -------------------------------
import trimesh
from pytorch3d.renderer import PerspectiveCameras


R_ref = c2w[:3, :3]
t_ref = c2w[:3, 3:4]

num_views = 10
candidate_views = []
for i in range(num_views):
    if i < num_views // 2:
        # 在主视角附近采样 (小扰动)
        noise_rot = trimesh.transformations.euler_matrix(
            *(np.random.normal(0, np.pi/36, 3))  # 大约±5°旋转
        )[:3, :3]
        R = noise_rot @ R_ref  # 小旋转扰动
        t = t_ref + np.random.normal(0, 0.05, size=(3, 1))  # 小平移扰动
    else:
        # 全局随机采样
        angles = np.random.uniform(0, np.pi, 3)
        R = trimesh.transformations.euler_matrix(*angles)[:3, :3]
        t = np.random.uniform(-1, 1, size=(3, 1))

    K = np.array([[focal_length, 0, center[0]],
                  [0, focal_length, center[1]],
                  [0, 0, 1]])
    candidate_views.append({"K": K, "R": R, "t": t})


# -------------------------------
# Step 3: 渲染并保存
# -------------------------------
os.makedirs("rendered_views", exist_ok=True)
results = []

for i, v in enumerate(candidate_views):
    K, R, t = v["K"], v["R"], v["t"]

    diff = compute_view_diff(points_np, K, R, t, ref_dir, image_w, image_h)

    cameras = PerspectiveCameras(
        device=device,
        R=torch.tensor(R[None], dtype=torch.float32, device=device),
        T=torch.tensor(t.T, dtype=torch.float32, device=device),
        focal_length=((K[0,0], K[1,1]),),
        principal_point=((K[0,2], K[1,2]),),
        image_size=((image_h, image_w),)
    )

    renderer = setup_renderer(cameras, (image_h, image_w))
    rendered = render_pcd(points_np, colors_np, cameras, renderer, device)
    img_rgb = rendered[0, ..., :3].cpu().numpy()

    psnr_val = psnr(ref_img, img_rgb, data_range=1.0)
    h, w = img_rgb.shape[:2]
    win_size = min(7, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
    ssim_val = ssim(ref_img, img_rgb, channel_axis=2, win_size=win_size, data_range=1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(f"View {i} | Diff={diff:.3f} | SSIM={ssim_val:.3f}", fontsize=10)
    plt.savefig(f"rendered_views/view_{i}.png", bbox_inches="tight")
    plt.close(fig)

    results.append({"view_id": i, "difficulty": diff, "psnr": psnr_val, "ssim": ssim_val})

df = pd.DataFrame(results)
df.to_csv("rendered_views/results.csv", index=False)
print("✅ 渲染完成，已保存到 rendered_views/")
