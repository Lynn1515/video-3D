import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pandas as pd
import cv2
from pytorch3d.structures import Pointclouds
from sklearn.neighbors import NearestNeighbors
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRenderer,
    PointLights,
    PointsRasterizer,
    PointsRasterizationSettings,
    AlphaCompositor
)

# -------------------------------
# 设备
# -------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 读取 PLY 点云
# -------------------------------
ply_file = "/home/cx_wchn/lnwang/ViewCrafter/test/pcd/pcd0.ply"
mesh = trimesh.load(ply_file)

points_np = np.asarray(mesh.vertices)   # [N,3]
colors_np = mesh.visual.vertex_colors[:, :3] / 255.0 \
    if mesh.visual.vertex_colors is not None else np.ones_like(points_np)

# 点云归一化
center = points_np.mean(axis=0)
points_np = points_np - center
scale = np.max(np.linalg.norm(points_np, axis=1))
points_np = points_np / scale

# 转 torch
points = torch.tensor(points_np, dtype=torch.float32, device=device)
colors = torch.tensor(colors_np, dtype=torch.float32, device=device)
pointcloud = Pointclouds(points=[points], features=[colors])

# -------------------------------
# 主相机（正面）
# -------------------------------
camera_distance = 0.88
R_ref = torch.tensor([[[-1, 0, 0],
                       [ 0, 1, 0],
                       [ 0, 0,-1]]], device=device, dtype=torch.float32)
T_ref = torch.tensor([[0, 0, camera_distance]], device=device, dtype=torch.float32)

R_ref = torch.tensor([[[-1, 0, 0],
                       [ 0, 1, 0],
                       [ 0, 0,-1]]], device=device, dtype=torch.float32)
T_ref = torch.tensor([[0, 0, camera_distance]], device=device, dtype=torch.float32)

# 相机内参 (像素坐标系)
image_h, image_w = 512, 512
focal_length = 500.0
principal_point = (image_w / 2, image_h / 2)

# -------------------------------
# 相机内参换算成 PyTorch3D 格式 (归一化)
# -------------------------------
f_ndc = focal_length / image_w
pp_ndc = ((principal_point[0] - image_w / 2) / image_w,
          (principal_point[1] - image_h / 2) / image_h)

# 内参矩阵 K (像素坐标系, 用来计算视角难度)
K = np.array([[focal_length, 0, principal_point[0]],
              [0, focal_length, principal_point[1]],
              [0, 0, 1]])

# -------------------------------
# 难度计算函数
# -------------------------------
def compute_difficulty_with_masks_torch(
    points,           # torch.Tensor (N,3) on device
    K,                # numpy (3,3) 或 torch.Tensor
    R, T,             # torch.Tensor R:(1,3,3), T:(1,3)
    R_ref, T_ref,     # torch.Tensor R_ref:(1,3,3), T_ref:(1,3)
    image_w, image_h,
    pointcloud=None,  # Pointclouds object (for渲染)
    save_dir=None,    # 如果不为 None，会保存对比图
    view_idx=0,       # 文件名索引
    alpha=1.0, beta=1.0, gamma=0.1,
    raster_radius=0.01,
    sobel_ksize=5,
    depth_grad_thresh=0.5,
    morph_op=None,
    morph_kernel_size=5,
    render_debug=True  # ✅ 控制是否渲染和保存可视化
):
    """
    计算候选视角难度，同时可选地渲染调试图像。
    返回:
        difficulty: float
        info: dict (包含子分数；若 render_debug=True，还包含 mask / depth / color)
    """
    device = points.device
    
    # --- 1) 基础难度计算 ---
    if not torch.is_tensor(K):
        K_t = torch.tensor(K, dtype=torch.float32, device=device)
    else:
        K_t = K.to(device)

    R0, T0 = R[0], T[0]
    Rref0 = R_ref[0]

    # 相机坐标系
    P_cam = (R0 @ points.T) + T0.view(3,1)
    P_camT = P_cam.T
    z = P_camT[:, 2]

    # 像素坐标
    uv_h = (K_t @ P_cam).T
    uv = uv_h[:, :2] / (uv_h[:, 2:3] + 1e-8)

    mask_vis = (uv[:,0] >= 0) & (uv[:,0] < image_w) & \
               (uv[:,1] >= 0) & (uv[:,1] < image_h) & (z > 0)
    vis_ratio = mask_vis.float().mean()
    vis_score = (1.0 - vis_ratio).item()

    if mask_vis.sum() > 0:
        depth_var = (z[mask_vis].var() / (z[mask_vis].mean()**2 + 1e-6)).item()
    else:
        depth_var = 1.0

    cam_dir = -R0[:, 2]
    cam_ref_dir = -Rref0[:, 2]
    cos_angle = torch.clamp(torch.dot(cam_dir, cam_ref_dir) /
                            (torch.norm(cam_dir) * torch.norm(cam_ref_dir) + 1e-8),
                            -1.0, 1.0).item()
    angle_score = (np.arccos(cos_angle) / np.pi)

    difficulty = alpha * vis_score + beta * depth_var + gamma * angle_score

    info = {
        'vis_score': vis_score,
        'depth_var': float(depth_var),
        'angle_score': float(angle_score),
        'difficulty': float(difficulty)
    }

    # --- 2) 可选渲染调试 ---
    if not render_debug or pointcloud is None:
        return float(difficulty), info

    # PyTorch3D camera 参数
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    f_ndc_x = fx / image_w
    f_ndc_y = fy / image_h
    pp_ndc = ((cx - image_w * 0.5) / image_w,
              (cy - image_h * 0.5) / image_h)

    cameras = PerspectiveCameras(
        device=device,
        R=R, T=T,
        focal_length=((f_ndc_x, f_ndc_y),),
        principal_point=(pp_ndc,),
        image_size=((image_h, image_w),)
    )

    raster_settings = PointsRasterizationSettings(
        image_size=(image_h, image_w),
        radius=raster_radius,
        points_per_pixel=10,
        bin_size=0
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

    fragments = rasterizer(pointcloud)
    zbuf = fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float32)

    # --- 动态获取最大值作为 "far" ---
    far_val = np.max(zbuf)
    print(f"[DEBUG] zbuf min={zbuf.min():.3f}, max={zbuf.max():.3e}")

    # 使用阈值检测无效深度
    #hole_mask = (zbuf >= far_val * 0.99).astype(np.uint8) * 255
    hole_mask = (zbuf <= 0.0).astype(np.uint8) * 255
    
    from pytorch3d.renderer import AlphaCompositor, PointsRenderer
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    color_img = renderer(pointcloud)[0, ..., :3].detach().cpu().numpy()

    depth_valid = np.where(np.isinf(zbuf), 0.0, zbuf)
    depth_norm = cv2.normalize(depth_valid.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

    depth_grad_x = cv2.Sobel(depth_norm, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    depth_grad_y = cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    depth_gradient = np.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)
    error_mask = (depth_gradient > depth_grad_thresh).astype(np.uint8) * 255

    if morph_op is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        if morph_op == 'open':
            hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)
            error_mask = cv2.morphologyEx(error_mask, cv2.MORPH_OPEN, kernel)
        elif morph_op == 'close':
            hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel)
            error_mask = cv2.morphologyEx(error_mask, cv2.MORPH_CLOSE, kernel)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        color_bgr = (color_img * 255.0).astype(np.uint8)[..., ::-1]
        hole_color = cv2.cvtColor(hole_mask, cv2.COLOR_GRAY2BGR)
        error_color = cv2.cvtColor(error_mask, cv2.COLOR_GRAY2BGR)
        concat = np.concatenate([color_bgr, hole_color, error_color], axis=1)
        cv2.imwrite(os.path.join(save_dir, f"view_{view_idx:03d}_comparison.png"), concat)

    info.update({
        'hole_mask': hole_mask,
        'error_mask': error_mask,
        'color': color_img,
        'depth': zbuf
    })

        # --- 统计并打印 ---

    hole_area = hole_mask.sum() / 255 / hole_mask.size 
    error_area = error_mask.sum() / 255 / error_mask.size
    difficulty_new = alpha * hole_area + beta * error_area + gamma * angle_score
    print(f"[DEBUG] view {view_idx}: "
          f"diff: {difficulty_new},"
          f"hole_area={hole_area:.4f}, "
          f"error_area={error_area:.4f},"
          f"angle_score={angle_score:.4f}")

    return float(difficulty_new), info



def compute_difficulty_fast_torch(
    points,        # (N,3) torch.Tensor
    K,             # (3,3) numpy or torch.Tensor
    R, T,          # R: (1,3,3), T: (1,3)
    R_ref, T_ref,  # 参考视角
    image_w, image_h,
    alpha=0.5, beta=0.3, gamma=0.2,
    depth_grad_thresh=0.5,
    neighbor_k=6
):
    """
    快速估算难度，无需渲染。
    返回:
        difficulty: float
        info: dict（包含hole近似、error近似、角度评分）
    """
    device = points.device
    R0, T0 = R[0], T[0]
    Rref0 = R_ref[0]

    # -------- (1) 角度分数（与参考视角夹角） --------
    cam_dir = -R0[:, 2]
    cam_ref_dir = -Rref0[:, 2]
    cos_angle = torch.clamp(torch.dot(cam_dir, cam_ref_dir) /
                            (torch.norm(cam_dir) * torch.norm(cam_ref_dir) + 1e-8),
                            -1.0, 1.0).item()
    angle_score = (np.arccos(cos_angle) / np.pi)

    # -------- (2) 投影可见性分数（近似 hole_area） --------
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # 相机坐标系下点云
    points_cam = (R0 @ points.T + T0.unsqueeze(-1)).T  # (N,3)
    z = points_cam[:, 2]
    xy = points_cam[:, :2] / z.unsqueeze(1)
    u = fx * xy[:, 0] + cx
    v = fy * xy[:, 1] + cy

    # 落在图像内部并且 z > 0
    valid_mask = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h) & (z > 0)
    projected_ratio = valid_mask.sum().item() / points.shape[0]
    approx_hole_area = 1.0 - projected_ratio

    # -------- (3) 几何 roughness 分数（近似 error_area） --------
    # z_vals = points[:, 2].detach().cpu().numpy()
    # points_np = points.detach().cpu().numpy()
    # nn = NearestNeighbors(n_neighbors=neighbor_k).fit(points_np)
    # _, indices = nn.kneighbors()
    # z_neighbors = z_vals[indices]
    # z_diff = np.abs(z_neighbors - z_vals[:, None])
    # high_grad_ratio = (z_diff > depth_grad_thresh).mean()
    # approx_error_area = high_grad_ratio

    z_vals = points[:, 2].detach().cpu().numpy()
    points_np = points.detach().cpu().numpy()
    nn = NearestNeighbors(n_neighbors=neighbor_k).fit(points_np)
    _, indices = nn.kneighbors()

    # 标准化后的 z 值（更鲁棒）
    z_mean = np.mean(z_vals)
    z_std = np.std(z_vals) + 1e-6
    z_norm = (z_vals - z_mean) / z_std
    z_neighbors = z_norm[indices]
    z_diff = np.abs(z_neighbors - z_norm[:, None])

    # 相对阈值判断（如 0.8 标准差以上差异）
    z_thresh = 0.8
    high_grad_ratio = (z_diff > z_thresh).mean()
    approx_error_area = high_grad_ratio

    # -------- 综合难度 --------
    difficulty = alpha * approx_hole_area + beta * approx_error_area + gamma * angle_score

    info = {
        "approx_hole_area": float(approx_hole_area),
        "approx_error_area": float(approx_error_area),
        "angle_score": float(angle_score),
    }
    print(f"[DEBUG] view Fast: "
          f"hole_area={approx_hole_area:.4f}, "
          f"error_area={approx_error_area:.4f}")
    
    return float(difficulty), info

def compute_difficulty_torch(points, K, R, T, R_ref, T_ref, image_w, image_h,
                             alpha=0.5, beta=0.3, gamma=0.2):
    """
    Torch 版本的候选视角难度计算函数
    输入:
        points: (N,3) torch.Tensor，稀疏点云，世界坐标系，可在 GPU 上
        K: (3,3) 内参矩阵，numpy 或 torch.Tensor
        R, T: torch.Tensor, 候选视角外参, R:(1,3,3), T:(1,3)
        R_ref, T_ref: torch.Tensor, 主视角外参, R_ref:(1,3,3), T_ref:(1,3)
        image_w, image_h: 图像尺寸
        alpha, beta, gamma: 权重
    输出:
        difficulty: float，值越大表示越难
    """
    device = points.device

    # 确保 K 为 torch.Tensor
    if not torch.is_tensor(K):
        K_t = torch.tensor(K, dtype=torch.float32, device=device)
    else:
        K_t = K.to(device)

    # -----------------------
    # 1️⃣ 可见性
    # -----------------------
    # 投影到相机坐标系
    P_cam = (R @ points.T + T.T).T  # (N,3)
    z = P_cam[:,2]
    
    # 投影到像素坐标
    uv_h = (K_t @ P_cam.T).T
    uv = uv_h[:, :2] / (uv_h[:, 2:3] + 1e-8)

    # 可见点掩码
    mask = (uv[:,0] >= 0) & (uv[:,0] < image_w) & \
           (uv[:,1] >= 0) & (uv[:,1] < image_h) & (z > 0)
    vis_score = 1.0 - mask.float().mean()  # 可见性差 → 难度高

    # -----------------------
    # 2️⃣ 深度不确定性
    # -----------------------
    if mask.sum() > 0:
        depth_var = torch.var(z[mask]) / (torch.mean(z[mask])**2 + 1e-6)
    else:
        depth_var = torch.tensor(1.0, device=device)

    # -----------------------
    # 3️⃣ 几何自相似性 / 视角夹角
    # -----------------------
    cam_dir = -R[0,:,2]
    cam_ref_dir = -R_ref[0,:,2]
    cos_angle = torch.clamp(torch.dot(cam_dir, cam_ref_dir) / 
                            (torch.norm(cam_dir) * torch.norm(cam_ref_dir) + 1e-8),
                            -1.0, 1.0)
    angle_score = torch.acos(cos_angle) / np.pi  # 归一化到 [0,1]

    # -----------------------
    # 总难度
    # -----------------------
    difficulty = alpha * vis_score + beta * depth_var + gamma * angle_score
    return difficulty.item()


# # -------------------------------
# # 构造候选视角
# # -------------------------------
# num_views = 10
# candidate_views = []
# candidate_views.append({"R": R_ref[0].cpu().numpy(), "T": T_ref[0].cpu().numpy()})  # 主视角

# for i in range(1, num_views):
#     # 小扰动旋转和平移
#     angles = np.random.normal(0, np.pi/36, 3)  # ~±5°
#     R_noise = trimesh.transformations.euler_matrix(*angles)[:3, :3]
#     R_new = R_noise @ R_ref[0].cpu().numpy()
#     T_new = T_ref[0].cpu().numpy() + np.random.normal(0, 0.05, size=(3,))
#     candidate_views.append({"R": R_new, "T": T_new})


# 构造环绕候选视角
# -------------------------------
num_views = 100
radius = camera_distance
candidate_views = []

# 主视角（正面）
candidate_views.append({"R": R_ref[0].cpu().numpy(), "T": T_ref[0].cpu().numpy()})

# 环绕视角，绕 Y 轴一圈
for i in range(1, num_views):
    theta = 2*np.pi*i/num_views  # 水平角度
    x = radius * np.sin(theta)
    y = 0.0                       # 固定高度
    z = radius * np.cos(theta)
    T_new = np.array([x, y, z])

    # 朝向原点
    forward = -T_new / np.linalg.norm(T_new)
    up = np.array([0,1,0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R_new = np.stack([right, up, forward], axis=1)

    candidate_views.append({"R": R_new, "T": T_new})
# -------------------------------
# 渲染所有视角
# -------------------------------
os.makedirs("rendered_views", exist_ok=True)
results = []

ref_dir = -R_ref[0].cpu().numpy()[:, 2]   # 主视角方向

for i, v in enumerate(candidate_views):
    R = torch.tensor(v["R"][None], dtype=torch.float32, device=device)
    T = torch.tensor(v["T"][None], dtype=torch.float32, device=device)

    cameras = PerspectiveCameras(
        device=device,
        R=R, T=T,
        focal_length=((f_ndc, f_ndc),),
        principal_point=(pp_ndc,),
        image_size=((image_h, image_w),)
    )

    # 每次新建 renderer
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.01,
        points_per_pixel=10
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor()
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # 渲染
    image = renderer(pointcloud)
    image_np = image[0, ..., :3].cpu().numpy()

    # 计算难度
    points_torch = torch.tensor(points_np, dtype=torch.float32, device=device)
    #diff = compute_difficulty_torch(points_torch, K, R, T, R_ref, T_ref, image_w, image_h,
    #                   alpha=0.5, beta=0.3, gamma=0.2)
    diff, _ = compute_difficulty_with_masks_torch(points_torch, K, R, T, R_ref, T_ref, image_w, image_h, pointcloud=pointcloud,
                                               save_dir="/home/cx_wchn/lnwang/ViewCrafter/rendered_views",view_idx=i)
    #diff,_ = compute_difficulty_fast_torch(points_torch, K, R, T, R_ref, T_ref, image_w, image_h, alpha=0.5, 
    #                                       beta=0.3, gamma=0.2, depth_grad_thresh=0.5, neighbor_k=6)
    #diff = compute_view_diff(points_np, K, v["R"], v["T"][:, None], ref_dir, image_w, image_h)

    # 保存
    plt.imsave(f"rendered_views/view_{i}.png", image_np)
    print(f"✅ 保存 rendered_views/view_{i}.png, range=({image_np.min():.3f},{image_np.max():.3f}), diff={diff:.3f}")

    results.append({"view_id": i, "difficulty": diff})


# 按 difficulty 排序
results_sorted = sorted(results, key=lambda x: x['difficulty'])

# 最简单的 10 个（difficulty 最小）
easiest_10 = results_sorted[:10]

# 最难的 10 个（difficulty 最大）
hardest_10 = results_sorted[-10:]

print("最简单的 10 个视角：")
for v in easiest_10:
    print(f"view_id={v['view_id']}, difficulty={v['difficulty']:.4f}")

print("\n最难的 10 个视角：")
for v in reversed(hardest_10):  # 从最难到稍难
    print(f"view_id={v['view_id']}, difficulty={v['difficulty']:.4f}")

# 保存结果表
df = pd.DataFrame(results)
df.to_csv("rendered_views/results.csv", index=False)
print("✅ 渲染完成，已保存到 rendered_views/")
