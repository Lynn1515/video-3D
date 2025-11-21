import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pandas as pd
import cv2
from pytorch3d.structures import Pointclouds
import torch.nn.functional as F
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
def compute_difficulty_with_masks_torch(
    points,           # torch.Tensor (N,3) on device
    K,                # numpy (3,3) 或 torch.Tensor
    R, T,             # torch.Tensor R:(1,3,3), T:(1,3)
    R_ref, T_ref,     # torch.Tensor R_ref:(1,3,3), T_ref:(1,3)
    image_w, image_h,
    pointcloud=None,  # Pointclouds object (for渲染)
    save_dir=None,    # 如果不为 None，会保存对比图
    view_idx=0,       # 文件名索引
    alpha=0.7, beta=1.0, gamma=0.1,
    raster_radius=0.01,
    sobel_ksize=5,
    depth_grad_thresh=0.35,
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
    uv = uv_h[:, :2] / (uv_h[:, 2:3] + 1e-12)

    mask_vis = (uv[:,0] >= 0) & (uv[:,0] < image_w) & \
               (uv[:,1] >= 0) & (uv[:,1] < image_h) & (z > 0)
    vis_ratio = mask_vis.float().mean()
    vis_score = (1.0 - vis_ratio).item()

    if mask_vis.sum() > 0:
        depth_var = (z[mask_vis].var() / (z[mask_vis].mean()**2 + 1e-12)).item()
    else:
        depth_var = 1.0

    cam_dir = -R0[:, 2]
    cam_ref_dir = -Rref0[:, 2]
    cos_angle = torch.clamp(torch.dot(cam_dir, cam_ref_dir) /
                            (torch.norm(cam_dir) * torch.norm(cam_ref_dir) + 1e-12),
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

    # cameras = PerspectiveCameras(
    #     device=device,
    #     R=R, T=T,
    #     focal_length=((f_ndc_x, f_ndc_y),),
    #     principal_point=(pp_ndc,),
    #     image_size=((image_h, image_w),)
    # )

    cameras = PerspectiveCameras(focal_length=fx, principal_point=((cx, cy),), in_ndc=False, 
                                                        image_size=((image_h, image_w),) , R=R, T=T, device=device)

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
    #print(f"[DEBUG] zbuf min={zbuf.min():.3f}, max={zbuf.max():.3e}")

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

    # ###########################敏感版
    gray = cv2.cvtColor((color_img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = cv2.GaussianBlur(np.abs(lap), (3,3), 0)

    artifact_mask1 = (lap_var > 30).astype(np.uint8) * 255

    #     # ---- 点稀疏检测 ----
    # idx_map = fragments.idx[0, ..., 0].detach().cpu().numpy()
    # point_count_map = (idx_map >= 0).astype(np.float32)
    # point_density = cv2.blur(point_count_map, (5,5))
    # sparse_mask = (point_density < 0.1).astype(np.uint8) * 255

    # # ---- 深度方差检测 ----
    # depth_std = cv2.blur(depth_norm**2, (5,5)) - cv2.blur(depth_norm, (5,5))**2
    # depth_std = np.sqrt(np.maximum(depth_std, 0))
    # var_mask = (depth_std > 0.02).astype(np.uint8) * 255

    # # ---- 合并 ----
    # error_mask = cv2.bitwise_or(error_mask, sparse_mask)
    # error_mask = cv2.bitwise_or(error_mask, var_mask)
        # ---- 合并 ----
    error_mask = cv2.bitwise_or(error_mask, artifact_mask1)

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
                            (torch.norm(cam_dir) * torch.norm(cam_ref_dir) + 1e-12),
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
    z_std = np.std(z_vals) + 1e-12
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
    R_refs, T_refs,          # (K,3,3), (K,3) torch.Tensor
    num_generate=2000,       # 初始采样点数
    desired_count=200,       # 最终采样数量
    radius=0.88,
    min_angle_deg=15.0,      # 与已知视角夹角下界
    max_angle_deg=40.0,      # 与已知视角夹角上界
    min_sep_deg=8.0,         # 候选点之间最小角度
    device='cuda'            # 设备
):
    """
    在已知视角的环形区域上采样候选相机位置（torch版）。
    R_ref, T_ref = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    返回:
        positions: (M,3) torch.Tensor
    """
    # 直接使用 T_refs 作为已知视角方向
    T_refs = T_refs.reshape(-1,3)
    known_views = T_refs / (T_refs.norm(dim=-1, keepdim=True) + 1e-12)  # (K,3)

    # Fibonacci 球面采样 (num_generate,3)
    i = torch.arange(0, num_generate, dtype=torch.float32, device=device) + 0.5
    phi = torch.acos(1 - 2*i/num_generate)               # polar angle
    theta = torch.pi * (1 + 5**0.5) * i                 # golden angle
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    pts = torch.stack([x,y,z], dim=1)                   # (num_generate,3)

    # 角度范围
    min_angle = torch.deg2rad(torch.tensor(min_angle_deg, device=device))
    max_angle = torch.deg2rad(torch.tensor(max_angle_deg, device=device))
    min_sep = torch.deg2rad(torch.tensor(min_sep_deg, device=device))

    # 过滤环形区域：必须落在至少一个已知视角夹角范围内
    dots = pts @ known_views.t()                        # (num_generate, K)
    angles = torch.acos(torch.clamp(dots, -1.0, 1.0))  # (num_generate, K)
    mask = ((angles >= min_angle) & (angles <= max_angle)).any(dim=1)
    cand = pts[mask]                                    # (Nc,3)
    if cand.shape[0] == 0:
        raise ValueError("No candidates found in the ring region, try relaxing angle range.")

    # 贪心选择保证最小角度分隔
    kept = []
    for v in cand:
        if len(kept) == 0:
            kept.append(v)
            if len(kept) >= desired_count:
                break
            continue
        kept_stack = torch.stack(kept, dim=0)         # (len_kept,3)
        angs = torch.acos(torch.clamp(kept_stack @ v, -1.0, 1.0))
        if torch.min(angs) >= min_sep:
            kept.append(v)
            if len(kept) >= desired_count:
                break

    if len(kept) < desired_count:
        kept = list(cand[:desired_count])

    positions = torch.stack(kept, dim=0) * radius      # (M,3)
    return positions


def sample_ring_candidates(
    R_refs, T_refs,          # (K,3,3), (K,3) torch.Tensor
    num_generate=2000,       # 初始采样点数
    desired_count=200,       # 最终采样数量
    radius=0.88,
    min_angle_deg=15.0,      # 与已知视角夹角下界
    max_angle_deg=40.0,      # 与已知视角夹角上界
    min_sep_deg=8.0,         # 候选点之间最小角度
    device='cuda'            # 设备
):
    """
    在已知视角的环形区域上采样候选相机位置（torch版）。
    返回:
        positions: (M,3) torch.Tensor
        belong_idx: (M,) torch.LongTensor, 表示每个采样点属于哪个已知视角环
    """
    T_refs = T_refs.reshape(-1,3)
    known_views = T_refs / (T_refs.norm(dim=-1, keepdim=True) + 1e-12)  # (K,3)

    # Fibonacci 球面采样
    i = torch.arange(0, num_generate, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2*i/num_generate)
    theta = torch.pi * (1 + 5**0.5) * i
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    pts = torch.stack([x,y,z], dim=1)  # (num_generate,3)

    # 角度约束
    min_angle = torch.deg2rad(torch.tensor(min_angle_deg, device=device))
    max_angle = torch.deg2rad(torch.tensor(max_angle_deg, device=device))
    min_sep = torch.deg2rad(torch.tensor(min_sep_deg, device=device))

    # 计算每个候选点与已知视角夹角
    dots = pts @ known_views.t()                     # (num_generate, K)
    angles = torch.acos(torch.clamp(dots, -1.0, 1.0))  # (num_generate, K)

    # 每个候选点属于哪个已知视角环
    mask = (angles >= min_angle) & (angles <= max_angle)  # (num_generate, K)
    valid_pts_idx = mask.any(dim=1)                        # 是否在任一环内
    cand = pts[valid_pts_idx]                              # 候选点
    cand_angles = angles[valid_pts_idx]                    # 候选点角度
    cand_mask = mask[valid_pts_idx]                        # 候选点mask

    if cand.shape[0] == 0:
        raise ValueError("No candidates found in the ring region, try relaxing angle range.")

    # 对每个候选点，取第一个满足条件的视角索引
    belong_idx = torch.argmax(cand_mask.to(torch.float32), dim=1)  # (Nc,)

    # 贪心选择保证最小角度分隔
    kept = []
    kept_idx = []
    for i, v in enumerate(cand):
        if len(kept) == 0:
            kept.append(v)
            kept_idx.append(belong_idx[i])
            if len(kept) >= desired_count:
                break
            continue
        kept_stack = torch.stack(kept, dim=0)  # (len_kept,3)
        angs_to_kept = torch.acos(torch.clamp(kept_stack @ v, -1.0, 1.0))
        if torch.min(angs_to_kept) >= min_sep:
            kept.append(v)
            kept_idx.append(belong_idx[i])
            if len(kept) >= desired_count:
                break

    # 不足时直接截取
    if len(kept) > desired_count:
        kept = list(cand[:desired_count])
        kept_idx = list(belong_idx[:desired_count])

    positions = torch.stack(kept, dim=0) * radius       # (M,3)
    belong_idx = torch.tensor(kept_idx, dtype=torch.long, device=device)  # (M,)
    #print("T_refs in sample", T_refs)
    #print("positions in sample", positions)
    return positions, belong_idx
# -----------------------------
# 将位置转换为朝向原点的 R,T（右，上，前列的矩阵）
# -----------------------------
def look_at_R_from_positions(positions, up_hint=None, device='cuda'):
    """
    positions: (M,3) torch.Tensor, camera centers in world coords
    返回:
        Rs: (M,3,3) torch.Tensor, 每个相机的旋转矩阵
        Ts: (M,3) torch.Tensor, 相机中心
    R: camera-to-world rotation, columns = [right, up, forward]
    forward = -normalize(position) (camera looks to origin)
    """
    if up_hint is None:
        up_hint = torch.tensor([0,1,0], dtype=torch.float32, device=device)
    else:
        up_hint = torch.tensor(up_hint, dtype=torch.float32, device=device)

    positions = positions.to(device)
    M = positions.shape[0]

    forward = -positions / (positions.norm(dim=-1, keepdim=True) + 1e-12)  # (M,3)

    # 检查与 up_hint 是否接近平行
    dot = torch.abs((forward * up_hint).sum(dim=-1))  # (M,)
    up = up_hint.unsqueeze(0).repeat(M,1)             # (M,3)
    mask = dot > 0.999
    if mask.any():
        up[mask] = torch.tensor([1,0,0], dtype=torch.float32, device=device)  # 替换接近平行的 up

    right = torch.cross(up, forward, dim=-1)
    right = right / (right.norm(dim=-1, keepdim=True) + 1e-12)
    up_corrected = torch.cross(forward, right, dim=-1)

    Rs = torch.stack([right, up_corrected, forward], dim=-1)  # (M,3,3)
    Ts = positions  # (M,3)

    return Rs, Ts

# def look_at_R_from_positions(positions, at=None, up=(0, 1, 0), device='cuda'):
#     """
#     positions: (M,3) 相机位置（世界坐标系）
#     at: 目标点（默认看向世界原点）
#     up: 上方向
#     返回:
#         Rs: (M,3,3) world-to-camera 旋转矩阵 (可直接用于 PyTorch3D)
#         Ts: (M,3)   world-to-camera 平移向量
#     """
#     if at is None:
#         at = torch.zeros(1, 3, device=device)
#     if isinstance(up, tuple) or isinstance(up, list):
#         up = torch.tensor(up, dtype=torch.float32, device=device)

#     positions = positions.to(device)
#     M = positions.shape[0]

#     # forward: 目标点 - 相机位置
#     forward = at.to(device) - positions   # (M,3)
#     forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-9)

#     # 右向量
#     right = torch.cross(up.expand_as(forward), forward, dim=-1)
#     right = right / (right.norm(dim=-1, keepdim=True) + 1e-9)

#     # 重新计算正交化后的 up
#     true_up = torch.cross(forward, right, dim=-1)

#     # === 拼成 camera-to-world (列向量) ===
#     c2w = torch.stack([right, true_up, forward], dim=-1)  # (M,3,3)

#     # === 转成 world-to-camera (PyTorch3D需求) ===
#     R = c2w.permute(0, 2, 1)        # 转置 = 逆
#     T = -torch.bmm(R, positions[..., None]).squeeze(-1)  # -R*C

#     return R, T


def select_simple_view_and_generate_trajectory(
    points,
    K,
    R_ref, T_ref,
    image_w, image_h,
    pointcloud,
    candidate_views,
    difficulty_fn=compute_difficulty_with_masks_torch,
    visited_views=None,       # (K,3) 已知相机中心向量
    difficulty_threshold=0.5,
    exclude_angle_deg=20.0,   # ⚠️ 新增参数：排除已知视角附近多少度内
    device="cuda",
    save_dir=None,
):
    """
    从候选视角中选择一个简单视角，并在其附近生成相机轨迹
    """
    results = []
    # 把 visited_views 转成单位向量 (K,3)
    visited_vecs = None
    # print("T_refs in detect", visited_views)
    # print("positions in sample", candidate_views)
    if visited_views is not None and len(visited_views) > 0:
        visited_vecs = visited_views / (visited_views.norm(dim=-1, keepdim=True) + 1e-12)
    exclude_angle = torch.deg2rad(torch.tensor(exclude_angle_deg, device=device))

    for i, v in enumerate(candidate_views):
        R = torch.tensor(v["R"][None], dtype=torch.float32, device=device)
        T = torch.tensor(v["T"][None], dtype=torch.float32, device=device)

        # 计算相机位置的方向向量（归一化）
        cam_pos = T.view(-1)          # (3,)
        cam_vec = cam_pos / (cam_pos.norm() + 1e-12)

        # ---- 过滤：如果和任一已知视角夹角 < exclude_angle，则跳过 ----
        if visited_vecs is not None:
            # angs_to_kept = torch.acos(torch.clamp(kept_stack @ v, -1.0, 1.0))
            #if torch.min(angs_to_kept) >= min_sep:
            dots = cam_vec @ visited_vecs.T      # (K,)
            angs = torch.acos(torch.clamp(dots, -1.0, 1.0))
            if torch.any(angs < exclude_angle):
                continue

        # c2w = torch.eye(4, dtype=torch.float32, device=device)
        # c2w[:3, :3] = R
        # c2w[:3, 3] = T

        # 求逆得到世界到相机变换 (PyTorch3D 标准)
        R_tmp = torch.stack([-R[:,:,0], -R[:,:,1], R[:,:,2]], dim=2)  # (N,3,3)
        T_tmp = T[:, :, None]  # (N,,1,3)
        new_c2w = torch.cat([R_tmp, T_tmp], dim=2)  # (N,3,4)
        hom = torch.tensor([[[0.,0.,0.,1.]]], device=device).repeat(new_c2w.shape[0],1,1)
        w2c = torch.linalg.inv(torch.cat((new_c2w, hom), dim=1))

        # R_tmp = torch.stack([-R[:,0], -R[:,1], R[:,2]], dim=1)  # (3,3)
        # new_c2w = torch.cat([R_tmp, T], dim=1)  # (N,3,4)
        # hom = torch.tensor([[0.,0.,0.,1.]], device=device)
        # w2c = torch.linalg.inv(torch.cat((new_c2w, hom), dim=0))
        R_w2c = w2c[:, :3, :3]#.permute(0,2,1)
        T_w2c = w2c[:, :3, 3]


        # w2c = torch.linalg.inv(c2w)
        # R_w2c = w2c[:3, :3][None]
        # T_w2c = w2c[:3, 3][None]

        # 计算难度
        diff, _ = difficulty_fn(
            points, K, R_w2c, T_w2c, R_ref, T_ref,
            image_w, image_h, pointcloud=pointcloud,
            save_dir=save_dir, view_idx=i
        )
        results.append({"view_id": i, "R": R_w2c, "T": T_w2c, "difficulty": diff, "belong_idx": v["belong_idx"]})

    if len(results) == 0:
        raise RuntimeError("没有可用候选视角（可能角度范围太严格，或候选视角全被排除了）")

    # 按难度排序
    results_sorted = sorted(results, key=lambda x: x['difficulty'])

    # 筛掉太难的，只留在阈值以下的
    filtered = [r for r in results_sorted if r['difficulty'] <= difficulty_threshold]
    if len(filtered) == 0:
        best_view = results_sorted[0]  # fallback: 最简单的
    else:
        best_view = filtered[0]

    print("+++++best view+++++", best_view["view_id"], "belong to", best_view["belong_idx"])
    return best_view


def interp_between_views(R1, T1, R2, T2, num_traj_points=25, traj_radius=0.05, device="cuda"):
    """
    在两个相机位置之间生成一个闭合环形轨迹，保证经过 T1 和 T2 (torch版)
    Args:
        R1, T1: 第一个相机外参 (3,3), (3,)
        R2, T2: 第二个相机外参 (3,3), (3,)
    Returns:
        traj_views: [{'R': (3,3), 'T': (3,)} ...]
    """
    T1 = T1.to(device).reshape(-1)
    T2 = T2.to(device).reshape(-1)

    # 主方向
    main_axis = T2 - T1
    main_axis = main_axis / (main_axis.norm() + 1e-12)

    # 构造正交方向
    tmp = torch.tensor([0., 1., 0.], device=device)
    if torch.abs(torch.dot(tmp, main_axis)) > 0.9:
        tmp = torch.tensor([1., 0., 0.], device=device)
    ortho1 = torch.cross(main_axis, tmp)
    ortho1 = ortho1 / (ortho1.norm() + 1e-12)
    ortho2 = torch.cross(main_axis, ortho1)
    ortho2 = ortho2 / (ortho2.norm() + 1e-12)

    traj_views = []
    for i in range(num_traj_points):
        t = torch.tensor(i / num_traj_points, device=device, dtype=torch.float32)
        # 沿主轴插值
        pos_base = (1 - t) * T1 + t * T2
        # 垂直扰动形成闭环
        theta = 2 * torch.pi * t
        offset = traj_radius * (torch.cos(theta) * ortho1 + torch.sin(theta) * ortho2)
        pos_new = pos_base + offset

        Rs, Ts = look_at_R_from_positions(pos_new[None])  # 这里要保证 look_at_R_from_positions 支持 torch
        traj_views.append({"R": Rs[0], "T": Ts[0]})

    return traj_views


def slerp(v0, v1, t):
    """球面线性插值 (v0,v1 已归一化), t 是标量"""
    v0 = F.normalize(v0, dim=0)
    v1 = F.normalize(v1, dim=0)
    dot = torch.clamp(torch.dot(v0, v1), -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if sin_theta < 1e-12:
        return v0
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return F.normalize(s0 * v0 + s1 * v1, dim=0)

def interp_spherical_between_views(R1, T1, R2, T2, n, device="cuda"):
    """
    在两个相机位置 (R1,T1), (R2,T2) 之间生成球面插值轨迹
    返回字典列表形式: [{"R": (3,3), "T": (3,)}]
    """
    T1, T2 = T1.to(device).view(-1), T2.to(device).view(-1)
    R1, R2 = R1.to(device), R2.to(device)

    t_vals = torch.linspace(0, 1, n, device=device)
    positions = torch.stack([slerp(T1, T2, t) for t in t_vals], dim=0)  # (n,3)

    up_hint = torch.tensor([0., 1., 0.], device=device)
    traj_views = []

    for pos in positions:
        pos = pos.view(-1)
        forward = -F.normalize(pos, dim=0)
        up = up_hint.clone()
        if torch.abs(torch.dot(forward, up)) > 0.999:
            up = torch.tensor([1., 0., 0.], device=device)
        right = F.normalize(torch.cross(up, forward), dim=0)
        up_corr = F.normalize(torch.cross(forward, right), dim=0)
        R = torch.stack([right, up_corr, forward], dim=1)
        traj_views.append({"R": R, "T": pos})

    return traj_views

def interp_linear_views(R1, T1, R2, T2, n, device="cuda"):
    """
    纯数值线性插值位置和旋转
    返回:
        traj_views: [{"R": (3,3), "T": (3,)}]
    """
    T1, T2 = T1.view(3).to(device), T2.view(3).to(device)
    R1, R2 = R1.view(3,3).to(device), R2.view(3,3).to(device)
    t_vals = torch.linspace(0, 1, n, device=device)

    traj_views = []
    for t in t_vals:
        T_new = (1-t)*T1 + t*T2
        R_new = (1-t)*R1 + t*R2
        traj_views.append({"R": R_new, "T": T_new})
    return traj_views

# def interp_between_views(R1, T1, R2, T2, num_traj_points=25, traj_radius=0.05):
#     """
#     在两个相机位置之间生成一个闭合环形轨迹，保证经过 T1 和 T2
#     """
#     T1 = to_numpy(T1)
#     T2 = to_numpy(T2)
    
#     # 主方向
#     main_axis = T2 - T1
#     main_axis = main_axis.reshape(-1) 
#     main_axis /= (np.linalg.norm(main_axis) + 1e-8)
    
#     # 构造正交方向
#     tmp = np.array([0,1,0], dtype=np.float32)
#     if abs(np.dot(tmp, main_axis)) > 0.9:
#         tmp = np.array([1,0,0], dtype=np.float32)
#     ortho1 = np.cross(main_axis, tmp)
#     ortho1 /= (np.linalg.norm(ortho1) + 1e-8)
#     ortho2 = np.cross(main_axis, ortho1)
#     ortho2 /= (np.linalg.norm(ortho2) + 1e-8)
    
#     traj_views = []
#     for i in range(num_traj_points):
#         t = i / num_traj_points
#         # 沿主轴插值
#         pos_base = (1-t)*T1 + t*T2
#         # 垂直扰动形成闭环
#         theta = 2 * np.pi * t
#         offset = traj_radius * (np.cos(theta)*ortho1 + np.sin(theta)*ortho2)
#         pos_new = pos_base + offset
#         Rs, Ts = look_at_R_from_positions(pos_new.reshape(1,3))
#         traj_views.append({"R": Rs[0], "T": Ts[0]})
    
#     return traj_views
