import sys
sys.path.append('./extern/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from test_diffculty_cal3 import interp_linear_views, compute_difficulty_with_masks_torch, sample_ring_candidates, look_at_R_from_positions, select_simple_view_and_generate_trajectory, interp_between_views
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
import glob
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis,image_guided_synthesis_w_ctrl2
from pathlib import Path
from torchvision.utils import save_image

class ViewCrafter:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        self.setup_dust3r()
        self.setup_diffusion()
        # initialize ref images, pcd
        if not gradio:
            if os.path.isfile(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images)
            elif os.path.isdir(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images, clean_pc = True)    
            else:
                print(f"{self.opts.image_dir} doesn't exist")           
        
    def run_dust3r(self, input_images,clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)#pts3d (2,288,512,3)  #view1#(2,288,512,3) #conf(2,288,512)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False):
        
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None

        return images, view_masks
    
    def run_render(self, pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views,renderer,self.device,nbv=False)
        return render_results, viewmask

    
    def run_diffusion(self, renderings):

        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            # [1,1,c,t,h,w]
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index)

            # save_results_seperate(batch_samples[0], self.opts.save_dir, fps=8)
            # torch.Size([1, 3, 25, 576, 1024]) [-1,1]

        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.) 

    def run_diffusion_batches(self, renderings):
        #torch.Size([2, 25, 576, 1024, 3])
        prompts = [self.opts.prompt] * renderings.shape[0]  # duplicate prompt per batch
        B, T, H, W, _ = renderings.shape
        videos = (renderings * 2. - 1.).permute(0, 4, 1, 2, 3).to(self.device)   # [B, 3, T, H, W]
        #videos = videos.unsqueeze(1).to(self.device) 
        #torch.Size([2, 3, 25, 576, 1024])
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            # [1,1,c,t,h,w]
            batch_samples = image_guided_synthesis_w_ctrl2(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index)

            # save_results_seperate(batch_samples[0], self.opts.save_dir, fps=8)
            # torch.Size([1, 3, 25, 576, 1024]) [-1,1]
        #batch_samples#(2,1,3,25,576,1024)
        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.), torch.clamp(batch_samples[1][0].permute(1,2,3,0), -1., 1.) 
    def nvs_single_view(self, gradio=False):
        # 最后一个view为 0 pose
        c2ws = self.scene.get_im_poses().detach()[1:] #(1,4,4)
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        print("princ points", principal_points)
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)
        
        masks = None

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0],num_candidates, self.device)
            _, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image( viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask0_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.d_x[0]*depth_avg/focals.item(),self.opts.d_y[0]*depth_avg/focals.item(),self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj,num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
        else:
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj,num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
            #raise KeyError(f"Invalid Mode: {self.opts.mode}")

        render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                render_results[-1] = self.img_ori
                
        save_video(render_results, os.path.join(self.opts.save_dir, 'render0.mp4'))
        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))

        return diffusion_results


    def nvs_single_view(self, gradio=False):
        # 最后一个view为 0 pose
        c2ws = self.scene.get_im_poses().detach()[1:] 
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] #(2,288,512,3) a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]#list (288,512)
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整
        # print(pcd)
        # print(pcd[0].shape)
        # print(len(pcd))
        # print(H,W)
        # print("focals", focals)
        # print("radius", radius)
        # print("princ points", principal_points)
        # print("princ points", principal_points.shape)
        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)
        
        masks = None

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0],num_candidates, self.device)
            _, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image( viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask0_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.d_x[0]*depth_avg/focals.item(),self.opts.d_y[0]*depth_avg/focals.item(),self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj,num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
        elif self.opts.mode == 'view_select_iterative':
            R_tmp, T_tmp = c2ws[:, :3, :3], c2ws[:, :3, 3:]
            R_tmp = torch.stack([-R_tmp[:,:,0], -R_tmp[:,:,1], R_tmp[:,:,2]], dim=2)  # (N,3,3)
            new_c2w = torch.cat([R_tmp, T_tmp], dim=2)  # (N,3,4)
            hom = torch.tensor([[[0.,0.,0.,1.]]], device=self.device).repeat(new_c2w.shape[0],1,1)
            w2c_ref = torch.linalg.inv(torch.cat((new_c2w, hom), dim=1))
            w2c_ref = torch.linalg.inv(torch.cat((new_c2w, hom), dim=1))
            R_ref_new = w2c_ref[:,:3,:3]#.permute(0,2,1)
            T_ref_new = w2c_ref[:,:3,3]
            R_ref, T_ref = c2ws[:,:3, :3], c2ws[:,:3, 3:]
            T_ref = T_ref.squeeze(2)
            # 内参矩阵 K (像素坐标系, 用来计算视角难度)

            pp = principal_points[0]  # shape (2,)
            K = torch.tensor([[focals, 0, pp[0]],
                  [0, focals, pp[1]],
                  [0, 0, 1]], dtype=torch.float32, device=principal_points.device)

            #print(radius, c2ws[:,2,3])
            positions, idx_ = sample_ring_candidates(R_ref, T_ref, num_generate=3000, desired_count=100, radius=c2ws[:,2,3], min_angle_deg=20.0, max_angle_deg=30.0, min_sep_deg=6.0)
            Rs, Ts = look_at_R_from_positions(positions)

            # # build c2w 4x4 matrices (batch)
            # c2w_new = torch.zeros((Rs.shape[0], 4, 4), dtype=torch.float32, device=self.device)
            # c2w_new[:, :3, :3] = Rs
            # c2w_new[:, :3, 3] = Ts
            # c2w_new[:, 3, :] = torch.tensor([0.,0.,0.,1.], device=self.device)

            # w2c = torch.linalg.inv(c2w_new)
            # R_new = w2c[:,:3,:3]#.permute(0,2,1)
            # T_new = w2c[:,:3,3]

            # 构造 candidate_views 列表
            candidate_views = []
            # 先把 reference view 放最前（与原代码一致）
            #candidate_views.append({"R": R_ref[0], "T": T_ref[0]})
            for i in range(positions.shape[0]):
                candidate_views.append({"R": Rs[i], "T": Ts[i], "belong_idx": idx_[i]})

            print(f"总共 {len(candidate_views)} 个候选视角，开始渲染和计算难度...")
            os.makedirs("rendered_views", exist_ok=True)
            pts = torch.cat([p for p in [pcd[-1]]], dim=0).view(-1, 3).to(self.device)
            col = torch.from_numpy(np.concatenate([p for p in [imgs[-1]]])).view(-1, 3).to(self.device)
            point_cloud = Pointclouds(points=[pts], features=[col])#.extend(views)
            best_view = select_simple_view_and_generate_trajectory(
                pts, K, R_ref_new, T_ref_new,
                W, H,
                point_cloud,
                candidate_views,
                difficulty_fn=compute_difficulty_with_masks_torch,  # 或 compute_difficulty_fast_torch
                visited_views=T_ref,
                difficulty_threshold=0.7,
                device=self.device,
                save_dir='/home/cx_wchn/lnwang/ViewCrafter/rendered_views/0'
            )
            #print("插值RT", R_ref_new[best_view["belong_idx"]], T_ref_new[best_view["belong_idx"]], best_view["R"], best_view["T"])
            traj_views = interp_linear_views(R_ref_new[best_view["belong_idx"]], T_ref_new[best_view["belong_idx"]], best_view["R"], best_view["T"], n=self.opts.video_length)
         
            Rs = torch.stack([torch.tensor(v["R"], dtype=torch.float32, device=self.device) for v in traj_views], dim=0)  # (N,3,3)
            Ts = torch.stack([torch.tensor(v["T"], dtype=torch.float32, device=self.device) for v in traj_views], dim=0)  # (N,3)
            num_views = Rs.shape[0]
            camera_traj = PerspectiveCameras(focal_length=focals, principal_point=principal_points, in_ndc=False, 
                                                        image_size=((H, W),) , R=Rs, T=Ts, device=self.device)

            # candidate_views 现在包含若干均匀分布在极角带内的球面候选视角

        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]], masks, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                render_results[-1] = self.img_ori
                
        save_video(render_results, os.path.join(self.opts.save_dir, 'render0.mp4'))
        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))

        return diffusion_results

    def nvs_two_view(self, gradio=False):
        # 最后一个view为 0 pose
        c2ws = self.scene.get_im_poses().detach()[1:] #(1,4,4)
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)
        
        masks = None

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0],num_candidates, self.device)
            _, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image( viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask0_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.d_x[0]*depth_avg/focals.item(),self.opts.d_y[0]*depth_avg/focals.item(),self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj1, camera_traj2, num_views = generate_two_trajs(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device, viz_traj=True, save_dir = self.opts.save_dir)
        else:
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj1, camera_traj2, num_views = generate_two_trajs(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
            #raise KeyError(f"Invalid Mode: {self.opts.mode}")

        # ========== render traj1 ==========
        render_results1, viewmask1 = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj1, num_views)
        render_results1 = F.interpolate(render_results1.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results1[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                render_results1[-1] = self.img_ori
        save_video(render_results1, os.path.join(self.opts.save_dir, 'render0.mp4'))
        # ==========render traj2 ==========
        render_results2, viewmask2 = self.run_render(
            [pcd[-1]], [imgs[-1]], masks, H, W, camera_traj2, num_views
        )
        render_results2 = F.interpolate(render_results2.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results2[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1] == 0. and theta[-1] == 0. and r[-1] == 0.:
                render_results2[-1] = self.img_ori
        save_video(render_results2, os.path.join(self.opts.save_dir, 'render1.mp4'))
        #save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd1.ply') , mask_pc=False, reduce_pc=False)
        #render_results = torch.cat([render_results1, render_results2], dim=0)

        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)
        diffusion_results0 = self.run_diffusion(render_results1)
        diffusion_results1 = self.run_diffusion(render_results2)
        
        save_video((diffusion_results0 + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))
        save_video((diffusion_results1 + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion1.mp4'))

        concat_vid = torch.cat([diffusion_results0, diffusion_results1], dim=2) 
        save_video((concat_vid + 1.0)/2.0, os.path.join(self.opts.save_dir, 'diffusion_concat.mp4'))
        return diffusion_results0


    def nvs_single_view_w_ctrl(self, gradio=False):
        # 最后一个view为 0 pose
        c2ws = self.scene.get_im_poses().detach()[1:] #(1,4,4)
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)
        
        masks = None

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0],num_candidates, self.device)
            _, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image(viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask0_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj, num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.d_x[0]*depth_avg/focals.item(),self.opts.d_y[0]*depth_avg/focals.item(),self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj1, camera_traj2, num_views = generate_dual_traj_txt2(c2ws, H, W, focals, principal_points, phi, theta, r,
                                                                            self.opts.video_length, self.device, traj_type="offset", viz_traj=True, save_dir = self.opts.save_dir)
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")
        # ========== render traj1 ==========
        render_results1, viewmask1 = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj1, num_views)
        render_results1 = F.interpolate(render_results1.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results1[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.:
                render_results1[-1] = self.img_ori
                
        save_video(render_results1, os.path.join(self.opts.save_dir, 'render0.mp4'))

        # ==========render traj2 ==========
        render_results2, viewmask2 = self.run_render(
            [pcd[-1]], [imgs[-1]], masks, H, W, camera_traj2, num_views
        )
        render_results2 = F.interpolate(render_results2.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results2[0] = self.img_ori
        if self.opts.mode == 'single_view_txt':
            if phi[-1] == 0. and theta[-1] == 0. and r[-1] == 0.:
                render_results2[-1] = self.img_ori
        #####test
        for i in range(len(render_results2)):
            render_results2[i] = self.img_ori
        save_video(render_results2, os.path.join(self.opts.save_dir, 'render1.mp4'))
        #save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd1.ply') , mask_pc=False, reduce_pc=False)
        #render_results = torch.cat([render_results1, render_results2], dim=0)
        render_results = torch.stack([render_results1, render_results2], dim=0)
        diffusion_results0, diffusion_results1 = self.run_diffusion_batches(render_results)

        # vid0 = (torch.clamp(batch_samples[0][0], -1., 1.) + 1.0) / 2.0  # (3, 25, H, W)
        # vid1 = (torch.clamp(batch_samples[1][0], -1., 1.) + 1.0) / 2.0


        save_video((diffusion_results0 + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))
        save_video((diffusion_results1 + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion1.mp4'))

        concat_vid = torch.cat([diffusion_results0, diffusion_results1], dim=2)  # 上下拼接，高度加倍
        save_video((concat_vid + 1.0)/2.0, os.path.join(self.opts.save_dir, 'diffusion_concat.mp4'))

        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)
        #save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))

        return concat_vid

    def nvs_sparse_view(self,iter):

        c2ws = self.scene.get_im_poses().detach()#第一轮迭代产生5个(5,4,4)
        principal_points = self.scene.get_principal_points().detach()#(5,2)
        focals = self.scene.get_focals().detach()#(5,1)
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]#list
        depth_avg = depth[0][H//2,W//2] #以ref图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## masks for cleaner point cloud
        self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
        masks = self.scene.get_masks()
        depth = self.scene.get_depthmaps()
        bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
        masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
        masks = to_numpy(masks_new)

        ## render, 从c2ws[0]即ref image对应的相机开始
        imgs = np.array(self.scene.imgs)
        #self.opts.video_length, self.device,
        if self.opts.mode == 'single_view_ref_iterative':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=0, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[0:1], H, W, focals[0:1], principal_points[0:1], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],
                                                            self.opts.d_x[0]*depth_avg.item()/focals[0:1].item(), self.opts.d_y[0]*depth_avg.item()/focals[0:1].item(), self.opts.video_length, self.device)
            render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = self.img_ori
        elif self.opts.mode == 'single_view_1drc_iterative':
            self.opts.elevation -= self.opts.d_theta[iter-1]
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],self.opts.video_length, self.device)
            render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.
        elif self.opts.mode == 'single_view_nbv':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[0], self.opts.d_phi[0], num_candidates, self.device)
            _, viewmask = self.run_render(pcd, imgs,masks, H, W, candidate_poses,num_candidates,nbv=True)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image(viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask{iter}_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]   
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv    
            render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. 
        elif self.opts.mode == 'view_select_iterative':#pcd(5,288,512,3)
            c2ws, pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            R_tmp, T_tmp = c2ws[:, :3, :3], c2ws[:, :3, 3:]
            R_tmp = torch.stack([-R_tmp[:,:,0], -R_tmp[:,:,1], R_tmp[:,:,2]], dim=2)  # (N,3,3)
            new_c2w = torch.cat([R_tmp, T_tmp], dim=2)  # (N,3,4)
            hom = torch.tensor([[[0.,0.,0.,1.]]], device=self.device).repeat(new_c2w.shape[0],1,1)
            w2c_ref = torch.linalg.inv(torch.cat((new_c2w, hom), dim=1))
            R_ref_new = w2c_ref[:,:3,:3]#.permute(0,2,1)
            T_ref_new = w2c_ref[:,:3,3]
            R_ref, T_ref = c2ws[:,:3, :3], c2ws[:,:3, 3:]
            R_ref_first, T_ref_first = c2ws[0:1,:3, :3], c2ws[0:1,:3, 3:]
            T_ref = T_ref.squeeze(2)
            # 内参矩阵 K (像素坐标系, 用来计算视角难度)
            print("根据该相机位置生成候选", T_ref_first)

            pp = principal_points[0]  # shape (2,)
            K = torch.tensor([[focals[0], 0, pp[0]],
                  [0, focals[0], pp[1]],
                  [0, 0, 1]], dtype=torch.float32, device=principal_points.device)

            #print(radius, c2ws[:,2,3])
            positions, belong_idx = sample_ring_candidates(R_ref_first, T_ref_first, num_generate=(iter + 1) * 3000, desired_count=300, radius=c2ws[:,2,3][0], min_angle_deg=20.0, max_angle_deg=30.0, min_sep_deg=6.0)
            Rs, Ts = look_at_R_from_positions(positions)
            print(belong_idx)

            # # build c2w 4x4 matrices (batch)
            # c2w_new = torch.zeros((Rs.shape[0], 4, 4), dtype=torch.float32, device=self.device)
            # c2w_new[:, :3, :3] = Rs
            # c2w_new[:, :3, 3] = Ts
            # c2w_new[:, 3, :] = torch.tensor([0.,0.,0.,1.], device=self.device)

            # w2c = torch.linalg.inv(c2w_new)
            # R_new = w2c[:,:3,:3]#.permute(0,2,1)
            # T_new = w2c[:,:3,3]

            # 构造 candidate_views 列表（和你原有结构兼容）
            candidate_views = []
            # 先把 reference view 放最前（与原代码一致）
            #candidate_views.append({"R": R_ref[0], "T": T_ref[0]})
            for i in range(positions.shape[0]):
                candidate_views.append({"R": Rs[i], "T": Ts[i], "belong_idx": belong_idx[i]})

            print(f"总共 {len(candidate_views)} 个候选视角，开始渲染和计算难度...")
            os.makedirs("rendered_views", exist_ok=True)
            # pts = torch.cat([p for p in [pcd[-1]]], dim=0).view(-1, 3).to(self.device)
            # col = torch.from_numpy(np.concatenate([p for p in [imgs[-1]]])).view(-1, 3).to(self.device)
            masks_j = [torch.from_numpy(m) for m in masks]
            pts = torch.cat([p[m] for p, m in zip(pcd, masks_j)]).to(self.device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks_j)])).to(self.device)

            point_cloud = Pointclouds(points=[pts], features=[col])#.extend(views)
            best_view = select_simple_view_and_generate_trajectory(
                pts, K, R_ref_new, T_ref_new,
                W, H,
                point_cloud,
                candidate_views,
                difficulty_fn=compute_difficulty_with_masks_torch,  # 或 compute_difficulty_fast_torch
                visited_views=T_ref,
                difficulty_threshold=0.7,
                device=self.device,
                save_dir=f'/home/cx_wchn/lnwang/ViewCrafter/rendered_views/{str(iter)}'
            )
            #print("插值RT", R_ref_new[best_view["belong_idx"]], T_ref_new[best_view["belong_idx"]], best_view["R"], best_view["T"])
            print("根据该相机位置插值", T_ref_new[0])
            #traj_views = interp_linear_views(R_ref_new[best_view["belong_idx"]], T_ref_new[best_view["belong_idx"]], best_view["R"], best_view["T"], n=self.opts.video_length)
            traj_views = interp_linear_views(R_ref_new[0], T_ref_new[0], best_view["R"], best_view["T"], n=self.opts.video_length)
            # Rs = torch.stack([torch.tensor(v["R"], dtype=torch.float32, device=self.device) for v in traj_views], dim=0)  # (N,3,3)
            # Ts = torch.stack([torch.tensor(v["T"], dtype=torch.float32, device=self.device) for v in traj_views], dim=0)  # (N,3)
            # num_views = Rs.shape[0]
            # camera_traj = PerspectiveCameras(focal_length=focals, principal_point=principal_points, in_ndc=False, 
            #                                             image_size=((H, W),) , R=Rs, T=Ts, device=self.device)

            Rs = torch.stack([torch.tensor(v["R"], dtype=torch.float32, device=self.device) for v in traj_views], dim=0)  # (N,3,3)
            Ts = torch.stack([torch.tensor(v["T"], dtype=torch.float32, device=self.device) for v in traj_views], dim=0)  # (N,3)
            num_views = Rs.shape[0]
            # --- 确保这些参数的维度正确 ---
            focal_length = focals[best_view["belong_idx"]].reshape(1, 1).repeat(num_views, 2)  # (N,2)
            principal_point = principal_points[best_view["belong_idx"]].reshape(1, 2).repeat(num_views, 1)  # (N,2)
            image_size = torch.tensor([[H, W]], dtype=torch.float32, device=self.device).repeat(num_views, 1)  # (N,2)

            camera_traj = PerspectiveCameras(
                focal_length=focal_length,
                principal_point=principal_point,
                in_ndc=False,
                image_size=image_size,
                R=Rs,
                T=Ts,
                device=self.device
            )
            
            render_results, viewmask = self.run_render(pcd, imgs, masks, H, W, camera_traj, num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
            #render_results[0] = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.
            render_results[0] = self.img_ori

        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        save_video(render_results, os.path.join(self.opts.save_dir, f'render{iter}.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd{iter}.ply') , mask_pc=True, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion{iter}.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results
    
    def nvs_sparse_view_interp(self):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]

        if len(self.images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = self.scene.get_masks()
            depth = self.scene.get_depthmaps()
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
            masks = to_numpy(masks_new)
            mask_pc = True

        imgs = np.array(self.scene.imgs)

        camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        for i in range(len(self.img_ori)):
            render_results[i*(self.opts.video_length - 1)] = self.img_ori[i]
        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

        diffusion_results = []
        print(f'Generating {len(self.img_ori)-1} clips\n')
        for i in range(len(self.img_ori)-1 ):
            print(f'Generating clip {i} ...\n')
            diffusion_results.append(self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)]))
        print(f'Finish!\n')
        diffusion_results = torch.cat(diffusion_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_single_view_eval(self):

        # get camera trajectory of the input frames
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        c2ws,pcd =  world_point_to_kth(poses=c2ws, points=torch.stack(pcd), k=0, device=self.device)
        camera_traj,num_views = generate_traj(c2ws, H, W, focals, principal_points, self.device)
        
        # estimate pcd again using only one ref image
        images_ref = [self.images[0], copy.deepcopy(self.images[0])]
        images_ref[1]['idx'] = 1
        self.run_dust3r(input_images=images_ref)
        pcd_ref = self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)[0].detach()
        img_ref = np.array(self.scene.imgs)[0]
        masks = None

        render_results, viewmask = self.run_render([pcd_ref], [img_ref],masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori[0]
        save_video(render_results, os.path.join(self.opts.save_dir, f'render_ref0.mp4'))
        diffusion_results = self.run_diffusion(render_results)

        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion_ref0.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_single_view_ref_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def single_view_best_select_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, 5):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_1drc_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_nbv(self):
        # lef and right
        # d_theta and a_phi 是搜索空间的顶点间隔
        all_results = []
        ## FIXME: hard coded
        sample_rate = 6
        max_itr = 3

        idx = 1 #初始包含1张ref image
        for itr in range(0, max_itr):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)
    
    def load_initial_images(self, image_dir):
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式
        images = load_images([image_dir], size=512,force_1024 = True)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1

        return images, img_ori

    def load_initial_dir(self, image_dir):

        image_files = glob.glob(os.path.join(image_dir, "*"))

        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images = load_images(image_files, size=512,force_1024 = True)

        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.) 

        return images, img_gts

    def run_gradio(self,i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed):
        self.opts.elevation = float(i2v_elevation)
        self.opts.center_scale = float(i2v_center_scale)
        self.opts.ddim_steps = i2v_steps
        self.gradio_traj = [float(i) for i in i2v_d_phi.split()],[float(i) for i in i2v_d_theta.split()],[float(i) for i in i2v_d_r.split()]
        seed_everything(i2v_seed)
        torch.cuda.empty_cache()
        img_tensor = torch.from_numpy(i2v_input_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = center_crop_image(img_tensor) #1,3,h,w
        images = get_input_dict(image_tensor_resized,idx = 0,dtype = torch.float32)
        images = [images, copy.deepcopy(images)]
        images[1]['idx'] = 1
        self.images = images
        self.img_ori = (image_tensor_resized.squeeze(0).permute(1,2,0) + 1.)/2.

        # self.images: torch.Size([1, 3, 288, 512]), [-1,1]
        # self.img_ori:  torch.Size([576, 1024, 3]), [0,1]
        # self.images, self.img_ori = self.load_initial_images(image_dir=i2v_input_image)
        self.run_dust3r(input_images=self.images)
        self.nvs_single_view(gradio=True)

        traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion0.mp4")
        
        return traj_dir, gen_dir