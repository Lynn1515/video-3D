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
ply_file = "/home/cx_wchn/lnwang/ViewCrafter/output/20251007_2036_fern/pcd0.ply"
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

    # depth_grad_x = cv2.Scharr(depth_norm, cv2.CV_64F, 1, 0)
    # depth_grad_y = cv2.Scharr(depth_norm, cv2.CV_64F, 0, 1)


    depth_gradient = np.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)
    error_mask = (depth_gradient > depth_grad_thresh).astype(np.uint8) * 255

    ###########################敏感版
    #局部纹理异常检测（高频噪声）
    gray = cv2.cvtColor((color_img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = cv2.GaussianBlur(np.abs(lap), (3,3), 0)

    artifact_mask1 = (lap_var > 30).astype(np.uint8) * 255

    # depth_norm1 = depth_norm.astype(np.float32)
    # depth_var = cv2.Laplacian(depth_norm1, cv2.CV_64F)
    # var_mask = (np.abs(depth_var) > 0.07).astype(np.uint8) * 255

    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(artifact_mask1)
    # artifact_mask4 = np.zeros_like(error_mask)
    # for i in range(1, num_labels):
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     if area < 30:
    #         artifact_mask4[labels == i] = 255


    # ---- 点稀疏检测 ----
    idx_map = fragments.idx[0, ..., 0].detach().cpu().numpy()
    point_count_map = (idx_map >= 0).astype(np.float32)
    point_density = cv2.blur(point_count_map, (5,5))
    sparse_mask = (point_density < 0.1).astype(np.uint8) * 255

    # ---- 深度方差检测 ----
    depth_std = cv2.blur(depth_norm**2, (5,5)) - cv2.blur(depth_norm, (5,5))**2
    depth_std = np.sqrt(np.maximum(depth_std, 0))
    var_mask = (depth_std > 0.02).astype(np.uint8) * 255

    gray = cv2.cvtColor((color_img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    color_grad = cv2.Laplacian(gray, cv2.CV_64F)
    #color_mask = (np.abs(color_grad) > 70).astype(np.uint8) * 255
    black_mask = (gray < 10).astype(np.uint8)
    color_grad_abs = np.abs(color_grad)
    color_mask = ((color_grad_abs > 70) & (black_mask == 1)).astype(np.uint8) * 255
    
    # ---- 合并 ----
    error_mask = cv2.bitwise_or(error_mask, artifact_mask1)
    #error_mask = cv2.bitwise_or(error_mask, artifact_mask4)
    #error_mask = cv2.bitwise_or(error_mask, sparse_mask)
    #error_mask = cv2.bitwise_or(error_mask, var_mask)
    #error_mask = cv2.bitwise_or(error_mask, color_mask)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # error_mask = cv2.morphologyEx(error_mask, cv2.MORPH_CLOSE, kernel)  # 填小洞
    # error_mask = cv2.dilate(error_mask, kernel, iterations=1)           # 扩展边缘

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

# # -------------------------------
# # 构造候选视角
# -----------------------------
# 工具：spherical fibonacci 采样（均匀分布）
# -----------------------------
def fibonacci_sphere(n):
    """
    返回 n 个在单位球面上的点 (x,y,z)，近似均匀分布。
    输出 shape: (n,3)
    """
    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/n)          # polar angle 0..pi
    theta = np.pi * (1 + 5**0.5) * i   # golden angle multiples
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts = np.stack([x,y,z], axis=1)
    return pts  # (n,3)

# -----------------------------
# 角度计算：两个方向向量的夹角（弧度/度）
# -----------------------------
def angle_between_vecs(u, v):
    # u,v: (...,3) numpy
    # return angle in radians
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-12)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    cos = np.clip((u * v).sum(axis=-1), -1.0, 1.0)
    return np.arccos(cos)

# -----------------------------
# 从球面点筛选并贪心保持最小角度分隔
# -----------------------------
def sample_sphere_candidates(
    num_generate=2000,    # 先生成多少球面点（越大越接近理想）
    desired_count=200,    # 希望最终保留多少 candidate（或可用 min_sep_deg 控制）
    radius=0.88,
    phi_min_deg=20.0,     # polar angle 下界（度），0 = 顶点（+z）
    phi_max_deg=80.0,     # polar angle 上界（度），pi = 底部（-z）
    min_sep_deg=8.0       # 最小角度分隔（度），候选间夹角 >= min_sep_deg 才保留
):
    """
    返回 (positions, kept_indices)：
      positions: (M,3) numpy array in world coords (x,y,z) with norm == radius
    说明：
      - polar angle phi measured from +z axis (0..180 deg).
      - phi_min/phi_max 以度为单位，控制采样带。
    """
    # 1) 先用 Fibonacci 产生大量均匀点
    pts = fibonacci_sphere(num_generate)  # unit vectors
    
    # 2) 计算极角 phi（0..pi），并按 phi 过滤到想要的带
    # phi = arccos(z)
    phis = np.arccos(np.clip(pts[:,2], -1.0, 1.0))  # rad
    phi_min = np.deg2rad(phi_min_deg)
    phi_max = np.deg2rad(phi_max_deg)
    mask = (phis >= phi_min) & (phis <= phi_max)
    cand = pts[mask]
    if cand.shape[0] == 0:
        raise ValueError("No candidates after phi band filtering: widen phi range or increase num_generate.")
    
    # 3) 贪心筛选保证最小角度隔离
    min_sep_rad = np.deg2rad(min_sep_deg)
    kept = []
    for v in cand:
        if len(kept) == 0:
            kept.append(v)
            if len(kept) >= desired_count:
                break
            continue
        # 计算与已保留向量的角度最小值
        angs = angle_between_vecs(np.stack(kept), v[None, :])  # shape (k,1)
        if np.min(angs) >= min_sep_rad:
            kept.append(v)
            if len(kept) >= desired_count:
                break
    # 如果由于 min_sep 太大导致数量不足，尝试放宽 min_sep（退化策略）
    if len(kept) < desired_count:
        # 放宽策略：逐步降低 min_sep_rad
        cur_min_sep = min_sep_rad
        while len(kept) < desired_count and cur_min_sep > 0:
            cur_min_sep *= 0.9
            kept = []
            for v in cand:
                if len(kept) == 0:
                    kept.append(v)
                    if len(kept) >= desired_count:
                        break
                    continue
                angs = angle_between_vecs(np.stack(kept), v[None, :])
                if np.min(angs) >= cur_min_sep:
                    kept.append(v)
                    if len(kept) >= desired_count:
                        break
        # 若仍不足，直接取前 desired_count（退而求其次）
        if len(kept) < desired_count:
            kept = list(cand[:desired_count])
    kept = np.stack(kept, axis=0)  # (M,3)
    # scale to radius
    positions = kept * radius
    return positions  # (M,3) numpy

def to_numpy(x):
    """确保输入是 numpy 数组"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def get_view_directions(Rs, Ts):
    """
    输入:
      Rs: (K,3,3) torch/numpy
      Ts: (K,3)   torch/numpy
    返回:
      dirs: (K,3) numpy, 相机朝向原点的方向向量
    """
    Rs = to_numpy(Rs)
    Ts = to_numpy(Ts)
    centers = Ts  # 这里假设 T 就是相机中心（world 坐标）
    dirs = -centers / (np.linalg.norm(centers, axis=-1, keepdims=True) + 1e-12)
    return dirs, centers

def sample_ring_candidates(
    R_refs, T_refs,    # (K,3,3), (K,3)
    num_generate=2000,  # 初始采样点数（控制密度）
    desired_count=200,  # 最终希望采样数量
    radius=0.88,
    min_angle_deg=15.0, # 与已知视角夹角下界
    max_angle_deg=40.0, # 与已知视角夹角上界
    min_sep_deg=8.0     # 候选点之间的最小角度间隔
):
    """
    在已知视角的环形区域上采样候选相机位置。
    参数：
      - known_views: (K,3) numpy，已知参考方向（相机位置方向，指向原点）
      - min_angle_deg, max_angle_deg: 环带范围（夹角度数）
      - num_generate: 初始候选点数量（越大越均匀）
    返回：
      positions: (M,3)，候选相机位置
    """
    # 先把 R,T 转成方向向量#np.array([0.,0.,1.]) #
    _, known_views = get_view_directions(R_refs, T_refs)  # (K,3)

    pts = fibonacci_sphere(num_generate)  # 在球面上均匀采样
    cand = []

    min_angle = np.deg2rad(min_angle_deg)
    max_angle = np.deg2rad(max_angle_deg)

    # 过滤：必须落在至少一个已知视角的环形范围内
    for v in pts:
        angs = angle_between_vecs(known_views, v[None,:])  # (K,1)
        if np.any((angs >= min_angle) & (angs <= max_angle)):
            cand.append(v)
    cand = np.array(cand)
    if cand.shape[0] == 0:
        raise ValueError("No candidates found in the ring region, try relaxing angle range.")

    # 贪心选择，保证最小分隔
    min_sep_rad = np.deg2rad(min_sep_deg)
    kept = []
    for v in cand:
        if len(kept) == 0:
            kept.append(v)
            if len(kept) >= desired_count:
                break
            continue
        angs = angle_between_vecs(np.stack(kept), v[None,:])
        if np.min(angs) >= min_sep_rad:
            kept.append(v)
            if len(kept) >= desired_count:
                break

    if len(kept) < desired_count:
        kept = list(cand[:desired_count])

    kept = np.stack(kept, axis=0)
    positions = kept * radius
    return positions


# -----------------------------
# 将位置转换为朝向原点的 R,T（右，上，前列的矩阵）
# -----------------------------
def look_at_R_from_positions(positions, up_hint=np.array([0,1,0], dtype=np.float32)):
    """
    positions: (M,3) numpy, camera centers in world coords
    返回 R (M,3,3), T (M,3)
    R is the camera-to-world rotation (3x3) where columns are [right, up, forward]
    We define forward = -normalize(position) (camera looks to origin).
    """
    Rs = []
    Ts = []
    for pos in positions:
        forward = -pos / (np.linalg.norm(pos) + 1e-12)  # camera forward (z axis)
        # handle near-colinear up
        up = up_hint.copy()
        if abs(np.dot(forward, up)) > 0.999:  # almost colinear
            up = np.array([1,0,0], dtype=np.float32)  # choose different up
        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right) + 1e-12)
        up_corrected = np.cross(forward, right)
        R = np.stack([right, up_corrected, forward], axis=1)  # 3x3
        Rs.append(R.astype(np.float32))
        Ts.append(pos.astype(np.float32))
    Rs = np.stack(Rs, axis=0)  # (M,3,3)
    Ts = np.stack(Ts, axis=0)  # (M,3)
    return Rs, Ts

# -----------------------------
# 使用示例（替换你当前环形生成部分）
# -----------------------------
# 参数示例（可调）
radius = camera_distance  # 例如 0.88
num_generate = 2000       # 先生成多少球面点（越大更均匀）
desired_count = 200       # 最终候选视角数
phi_min_deg = 20.0        # 不要太俯视（最小极角）
phi_max_deg = 80.0        # 不要太仰视（最大极角）
min_sep_deg = 8.0         # 候选视角之间至少相差多少度

# positions = sample_sphere_candidates(
#     num_generate=num_generate,
#     desired_count=desired_count,
#     radius=radius,
#     phi_min_deg=phi_min_deg,
#     phi_max_deg=phi_max_deg,
#     min_sep_deg=min_sep_deg
# )  # (M,3) numpy


positions = sample_ring_candidates(
    R_refs=R_ref,
    T_refs=T_ref,
    num_generate=1000,
    desired_count=100,
    radius=0.46,
    min_angle_deg=20,
    max_angle_deg=30,
    min_sep_deg=8
)


Rs, Ts = look_at_R_from_positions(positions)
# 构造 candidate_views 列表（和你原有结构兼容）
candidate_views = []
# 先把 reference view 放最前（与原代码一致）
candidate_views.append({"R": R_ref[0].cpu().numpy(), "T": T_ref[0].cpu().numpy()})
for i in range(positions.shape[0]):
    candidate_views.append({"R": Rs[i], "T": Ts[i]})
# candidate_views 现在包含若干均匀分布在极角带内的球面候选视角

# -------------------------------

def select_simple_view_and_generate_trajectory(
    points,
    K,
    R_ref, T_ref,
    image_w, image_h,
    pointcloud,
    candidate_views,
    difficulty_fn=compute_difficulty_with_masks_torch,
    num_traj_points=10,
    traj_radius=0.05,
    visited_views=None,
    difficulty_threshold=0.5,
    device="cuda",
    save_dir=None,
):
    """
    从候选视角中选择一个简单视角，并在其附近生成相机轨迹

    Args:
        points: torch.Tensor (N,3) 点云
        K: 相机内参 (3,3)
        R_ref, T_ref: 参考相机外参
        image_w, image_h: 分辨率
        pointcloud: pytorch3d Pointclouds
        candidate_views: list of dict [{"R":, "T":}]
        difficulty_fn: callable，难度计算函数
        num_traj_points: 轨迹点数
        traj_radius: 环绕轨迹半径
        visited_views: 已访问过的视角id (set)
        difficulty_threshold: 难度阈值
        save_dir: 可选，保存可视化
        alpha, beta, gamma: 难度权重
    Returns:
        best_view: dict {"R","T","difficulty"}
        traj_views: list of dict
    """
    results = []
    for i, v in enumerate(candidate_views):
        if visited_views is not None and i in visited_views:
            continue

        R = torch.tensor(v["R"][None], dtype=torch.float32, device=device)
        T = torch.tensor(v["T"][None], dtype=torch.float32, device=device)

        diff, _ = difficulty_fn(
            points, K, R, T, R_ref, T_ref,
            image_w, image_h,pointcloud=pointcloud,
            save_dir=save_dir,view_idx=i
        )
        results.append({"view_id": i, "R": v["R"], "T": v["T"], "difficulty": diff})

    if len(results) == 0:
        raise RuntimeError("没有可用候选视角（可能 visited_views 覆盖全部）")

    # 按难度排序
    results_sorted = sorted(results, key=lambda x: x['difficulty'])

    # 筛掉太难的，只留在阈值以下的
    filtered = [r for r in results_sorted if r['difficulty'] <= difficulty_threshold]
    if len(filtered) == 0:
        best_view = results_sorted[0]  # fallback: 最简单的
    else:
        best_view = filtered[0]

    # ===== 在选中视角附近生成轨迹 =====
    traj_views = []
    R_best = best_view["R"]
    T_best = best_view["T"]

    # 用球面扰动：假设相机位置在球面上
    center = np.linalg.norm(T_best)
    pos_best = T_best / (np.linalg.norm(T_best) + 1e-8) * center

    for i in range(num_traj_points):
        theta = 2 * np.pi * i / num_traj_points
        offset = traj_radius * np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
        pos_new = pos_best + offset

        Rs, Ts = look_at_R_from_positions(pos_new[None, :])
        traj_views.append({"R": Rs[0], "T": Ts[0]})

    return best_view, traj_views

def interp_between_views(R1, T1, R2, T2, num_traj_points=25, traj_radius=0.05):
    """
    在两个相机位置之间生成一个闭合环形轨迹，保证经过 T1 和 T2
    """
    T1 = to_numpy(T1)
    T2 = to_numpy(T2)
    
    # 主方向
    main_axis = T2 - T1
    main_axis = main_axis.reshape(-1) 
    main_axis /= (np.linalg.norm(main_axis) + 1e-8)
    
    # 构造正交方向
    tmp = np.array([0,1,0], dtype=np.float32)
    if abs(np.dot(tmp, main_axis)) > 0.9:
        tmp = np.array([1,0,0], dtype=np.float32)
    ortho1 = np.cross(main_axis, tmp)
    ortho1 /= (np.linalg.norm(ortho1) + 1e-8)
    ortho2 = np.cross(main_axis, ortho1)
    ortho2 /= (np.linalg.norm(ortho2) + 1e-8)
    
    traj_views = []
    for i in range(num_traj_points):
        t = i / num_traj_points
        # 沿主轴插值
        pos_base = (1-t)*T1 + t*T2
        # 垂直扰动形成闭环
        theta = 2 * np.pi * t
        offset = traj_radius * (np.cos(theta)*ortho1 + np.sin(theta)*ortho2)
        pos_new = pos_base + offset
        Rs, Ts = look_at_R_from_positions(pos_new.reshape(1,3))
        traj_views.append({"R": Rs[0], "T": Ts[0]})
    
    return traj_views



# 渲染所有视角
# -------------------------------
print(f"总共 {len(candidate_views)} 个候选视角，开始渲染和计算难度...")
os.makedirs("rendered_views", exist_ok=True)
results = []

ref_dir = -R_ref[0].cpu().numpy()[:, 2]   # 主视角方向

best_view, _ = select_simple_view_and_generate_trajectory(
    points, K, R_ref, T_ref,
    image_w, image_h,
    pointcloud,
    candidate_views,
    difficulty_fn=compute_difficulty_with_masks_torch,  # 或 compute_difficulty_fast_torch
    num_traj_points=12,
    traj_radius=0.03,
    visited_views=None,
    difficulty_threshold=0.4,
    device=device,
    save_dir="/home/cx_wchn/lnwang/ViewCrafter/rendered_views/test"
)


# traj_views = interp_between_views(R_ref, T_ref, best_view["R"], best_view["T"])


# for i, v in enumerate(traj_views):
#     R = torch.tensor(v["R"][None], dtype=torch.float32, device=device)
#     T = torch.tensor(v["T"][None], dtype=torch.float32, device=device)

#     cameras = PerspectiveCameras(
#         device=device,
#         R=R, T=T,
#         focal_length=((f_ndc, f_ndc),),
#         principal_point=(pp_ndc,),
#         image_size=((image_h, image_w),)
#     )

#     # 每次新建 renderer
#     raster_settings = PointsRasterizationSettings(
#         image_size=512,
#         radius=0.01,
#         points_per_pixel=10
#     )
#     rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
#     compositor = AlphaCompositor()
#     renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

#     # 渲染
#     image = renderer(pointcloud)
#     image_np = image[0, ..., :3].cpu().numpy()

#     # 计算难度
#     points_torch = torch.tensor(points_np, dtype=torch.float32, device=device)
#     #diff = compute_difficulty_torch(points_torch, K, R, T, R_ref, T_ref, image_w, image_h,
#     #                   alpha=0.5, beta=0.3, gamma=0.2)
#     diff, _ = compute_difficulty_with_masks_torch(points_torch, K, R, T, R_ref, T_ref, image_w, image_h, pointcloud=pointcloud,
#                                                save_dir="/home/cx_wchn/lnwang/ViewCrafter/rendered_views/test",view_idx=i+1000)

#     # 保存
#     plt.imsave(f"rendered_views/test/view_{i}.png", image_np)
#     print(f"✅ 保存 rendered_views/test/view_{i}.png, range=({image_np.min():.3f},{image_np.max():.3f}), diff={diff:.3f}")

#     results.append({"view_id": i, "difficulty": diff})


# # 按 difficulty 排序
# results_sorted = sorted(results, key=lambda x: x['difficulty'])

# # 最简单的 10 个（difficulty 最小）
# easiest_10 = results_sorted[:10]

# # 最难的 10 个（difficulty 最大）
# hardest_10 = results_sorted[-10:]

# print("最简单的 10 个视角：")
# for v in easiest_10:
#     print(f"view_id={v['view_id']}, difficulty={v['difficulty']:.4f}")

# print("\n最难的 10 个视角：")
# for v in reversed(hardest_10):  # 从最难到稍难
#     print(f"view_id={v['view_id']}, difficulty={v['difficulty']:.4f}")

# # 保存结果表
# df = pd.DataFrame(results)
# df.to_csv("rendered_views/results.csv", index=False)
# print("✅ 渲染完成，已保存到 rendered_views/")
