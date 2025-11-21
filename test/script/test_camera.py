import trimesh
import torch
import numpy as np
import os
import math
import torchvision
import scipy
from tqdm import tqdm
import cv2  # Assuming OpenCV is used for image saving
from PIL import Image
import pytorch3d
import random
from PIL import ImageGrab
torchvision
from torchvision.utils import save_image
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import imageio
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import copy
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys
sys.path.append('./extern/dust3r')
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision.transforms import CenterCrop, Compose, Resize

def save_video(data,images_path,folder=None):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder]*len(data)
        images = [np.array(Image.open(os.path.join(folder_name,path))) for folder_name,path in zip(folder,data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=8, video_codec='h264', options={'crf': '10'})

def txt_interpolation(input_list,n,mode = 'smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew

def visualizer_frame(camera_poses, highlight_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 获取camera_positions[2]的最大值和最小值
    z_values = [pose[:3, 3][2] for pose in camera_poses]
    z_min, z_max = min(z_values), max(z_values)

    # 创建一个颜色映射对象
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["#00008B", "#ADD8E6"])
    # cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, pose in enumerate(camera_poses):
        camera_positions = pose[:3, 3]
        color = "blue" if i == highlight_index else "blue"
        size = 100 if i == highlight_index else 25
        color = sm.to_rgba(camera_positions[2])  # 根据camera_positions[2]的值映射颜色
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
            s=size,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Camera trajectory")
    ax.view_init(90+30, -90)

    plt.ylim(-0.1,0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    # new_width = int(width * 0.6)
    # start_x = (width - new_width) // 2 + new_width // 5
    # end_x = start_x + new_width
    # img = img[:, start_x:end_x, :]
    
    
    plt.close()

    return img
def generate_dual_traj_txt2(
    c2ws_anchor, H, W, fs, c,
    phi, theta, r,
    frame, device,
    traj_type="perturb",  # 新增参数
    viz_traj=False, save_dir=None
):
    """
    Generate two camera trajectories:
    - The original trajectory
    - A modified one based on selected traj_type:
        - 'perturb': small smooth noise (default)
        - 'left-right-sym': θ mirrored trajectory
        - 'up-down-sym': φ mirrored trajectory
    """
    scale_phi = 3.0
    scale_theta = 3.0
    scale_r = 0.01

    def interp(values, mode='smooth'):
        if len(values) > 3:
            out = txt_interpolation(values, frame, mode=mode)
            out[0], out[-1] = values[0], values[-1]
        else:
            out = txt_interpolation(values, frame, mode='linear')
        return out

    # ===== 1. 原始轨迹插值 =====
    phis = interp(phi)
    thetas = interp(theta)
    rs = interp(r)
    rs = rs * c2ws_anchor[0, 2, 3].cpu().numpy()

    # ===== 2. 构造第二条轨迹（根据不同模式）=====
    coarse_steps = 8
    coarse_idx = np.linspace(0, len(phis) - 1, coarse_steps).astype(int)

    phi_coarse = np.array(phis)[coarse_idx]
    theta_coarse = np.array(thetas)[coarse_idx]
    r_coarse = np.array(rs)[coarse_idx]

    if traj_type == "perturb":
        # 原始扰动模式
        phi_coarse += np.random.normal(scale=scale_phi, size=coarse_steps)
        theta_coarse += np.random.normal(scale=scale_theta, size=coarse_steps)
        r_coarse += np.random.normal(scale=scale_r * rs[0], size=coarse_steps)

    elif traj_type == "left-right-sym":
        # θ 左右对称
        theta_coarse = -theta_coarse  # 镜像
        # 其他保持不变

    elif traj_type == "up-down-sym":
        # φ 上下对称
        phi_coarse = -phi_coarse

    else:
        raise ValueError(f"Unknown traj_type: {traj_type}")

    # ===== 3. 插值新轨迹 =====
    phis_perturbed = interp(phi_coarse, mode='smooth')
    thetas_perturbed = interp(theta_coarse, mode='smooth')
    rs_perturbed = interp(r_coarse, mode='smooth')

    # ===== 4. 构造相机位姿 =====
    def build_c2ws(theta_list, phi_list, r_list):
        c2ws_list = []
        for th, ph, r_val in zip(theta_list, phi_list, r_list):
            c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r_val), device)
            c2ws_list.append(c2w_new)
        return torch.cat(c2ws_list, dim=0)

    c2ws_orig = build_c2ws(thetas, phis, rs)
    c2ws_perturb = build_c2ws(thetas_perturbed, phis_perturbed, rs_perturbed)

    # ===== 5. 转换为相机类 =====
    def convert_to_camera(c2ws):
        R, T = c2ws[:, :3, :3], c2ws[:, :3, 3:]
        R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)  # RDF -> LUF
        new_c2w = torch.cat([R, T], 2)
        w2c = torch.linalg.inv(torch.cat(
            (new_c2w, torch.Tensor([[[0, 0, 0, 1]]]).to(device).repeat(new_c2w.shape[0], 1, 1)), 1
        ))
        R_new, T_new = w2c[:, :3, :3].permute(0, 2, 1), w2c[:, :3, 3]
        image_size = ((H, W),)
        return PerspectiveCameras(
            focal_length=fs,
            principal_point=c,
            in_ndc=False,
            image_size=image_size,
            R=R_new,
            T=T_new,
            device=device
        )

    cameras1 = convert_to_camera(c2ws_orig)
    cameras2 = convert_to_camera(c2ws_perturb)

    # ===== 6. 可视化轨迹（可选）=====
    if viz_traj and save_dir is not None:
        poses1 = c2ws_orig.cpu().numpy()
        poses2 = c2ws_perturb.cpu().numpy()

        frames1 = [visualizer_frame(poses1, i) for i in range(len(poses1))]
        save_video(np.array(frames1) / 255., os.path.join(save_dir, 'viz_traj_orig.mp4'))

        frames2 = [visualizer_frame(poses2, i) for i in range(len(poses2))]
        save_video(np.array(frames2) / 255., os.path.join(save_dir, 'viz_traj_perturb.mp4'))

    return cameras1, cameras2, c2ws_orig.shape[0]



camera_traj1, camera_traj2, num_views = generate_dual_traj_txt2(c2ws, H, W, focals, principal_points, phi, theta, r,
                                                                            self.opts.video_length, self.device, traj_type="offset", viz_traj=True, save_dir = self.opts.save_dir)