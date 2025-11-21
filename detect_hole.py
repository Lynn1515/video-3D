import open3d as o3d
import numpy as np
import cv2
import os

def render_from_spherical_views(point_cloud_path, save_dir, width=640, height=480, n_views=36, elevation_deg=30,
                                 morph_op=None, morph_kernel_size=3):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载点云
    pcd = o3d.io.read_point_cloud(point_cloud_path)

    # 2. 创建离屏渲染器
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # 白背景
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    renderer.scene.add_geometry("pcd", pcd, mat)

    # 获取点云范围
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    radius = np.linalg.norm(extent) * 0.8  # 更靠近以增强深度分辨率

    # 3. 从球面上采样多个摄像机位置（固定仰角，绕中心旋转）
    elevation_rad = np.deg2rad(elevation_deg)
    views = []
    for i in range(n_views):
        azimuth_rad = 2 * np.pi * i / n_views
        cam_x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        cam_y = radius * np.sin(elevation_rad)
        cam_z = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        eye = np.array([cam_x, cam_y, cam_z]) + center
        up = np.array([0, 1, 0])  # 简单处理，假设y轴朝上
        views.append((eye, up))

    # 4. 遍历每个视角并渲染
    for idx, (eye, up) in enumerate(views):
        renderer.setup_camera(
            vertical_field_of_view=60.0,
            center=center.astype(np.float32),
            eye=eye.astype(np.float32),
            up=up.astype(np.float32)
        )

        # 渲染颜色图和深度图
        color = renderer.render_to_image()
        depth = renderer.render_to_depth_image(z_in_view_space=True)

        color_np = np.asarray(color)
        depth_np = np.asarray(depth)

        # 调试信息：查看深度值范围
        print(f"[view {idx}] depth min={depth_np.min():.4f}, max={depth_np.max():.4f}")

        # 生成空洞mask（基于 inf 判定）
        mask = np.isinf(depth_np).astype(np.uint8) * 255

        # 检测深度异常区域（误重建、高噪区域）
        depth_valid = np.where(np.isinf(depth_np), 0, depth_np)
        depth_valid = cv2.normalize(depth_valid, None, 0, 1.0, cv2.NORM_MINMAX)
        depth_grad_x = cv2.Sobel(depth_valid, cv2.CV_64F, 1, 0, ksize=5)
        depth_grad_y = cv2.Sobel(depth_valid, cv2.CV_64F, 0, 1, ksize=5)
        depth_gradient = np.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)
        error_mask = (depth_gradient > 0.5).astype(np.uint8) * 255

        # 可选：形态学操作
        if morph_op is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            if morph_op == 'open':
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                error_mask = cv2.morphologyEx(error_mask, cv2.MORPH_OPEN, kernel)
            elif morph_op == 'close':
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                error_mask = cv2.morphologyEx(error_mask, cv2.MORPH_CLOSE, kernel)

        # 保存三图横向拼接对比图
        color_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        error_color = cv2.cvtColor(error_mask, cv2.COLOR_GRAY2BGR)
        concat = np.concatenate([color_bgr, mask_color, error_color], axis=1)

        cv2.imwrite(os.path.join(save_dir, f'view_{idx:03d}_comparison.png'), concat)
        print(f"[view {idx}] 渲染图对比图已保存.")

# 示例调用：增加形态学开操作
render_from_spherical_views("your_point_cloud.ply", "./render_output", morph_op='open', morph_kernel_size=3)
