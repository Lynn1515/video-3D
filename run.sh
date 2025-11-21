python inference.py \
--image_dir /home/cx_wchn/lnwang/ViewCrafter/test/images/horns.jpg \
--out_dir ./output \
--traj_txt test/trajs/loop2.txt \
--mode 'view_select_iterative' \
--center_scale 1. \
--elevation=5 \
--seed 123 \
--d_theta -30  \
--d_phi 45 \
--d_r -.2   \
--d_x 50   \
--d_y 25   \
--ckpt_path ./checkpoints/model.ckpt \
--config configs/inference_pvd_1024.yaml \
--ddim_steps 50 \
--video_length 25 \
--device 'cuda:0' \
--height 576 --width 1024 \
--model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir /home/cx_wchn/lnwang/ViewCrafter/test/images/boy.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'view_select_iterative' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir /home/cx_wchn/lnwang/ViewCrafter/test/images/fern.JPG \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'view_select_iterative' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir /home/cx_wchn/lnwang/ViewCrafter/test/images/horns.jpg \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'view_select_iterative' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir /home/cx_wchn/lnwang/ViewCrafter/test/images/r10k_hearts_left.jpg \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'view_select_iterative' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/chair.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'view_select_iterative' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir /home/cx_wchn/lnwang/ViewCrafter/test/images/r10k_train_left.jpg \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'view_select_iterative' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir /home/cx_wchn/lnwang/ViewCrafter/test/images/truck.jpg \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'view_select_iterative' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/chair.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/espresso.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/ficus.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/materials.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/mic.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/pinecone.JPG \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/ship.png \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# python inference.py \
# --image_dir test/images/vasedeck.JPG \
# --out_dir ./output \
# --traj_txt test/trajs/loop2.txt \
# --mode 'single_view_txt' \
# --center_scale 1. \
# --elevation=5 \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \
# --ckpt_path ./checkpoints/model.ckpt \
# --config configs/inference_pvd_1024.yaml \
# --ddim_steps 50 \
# --video_length 25 \
# --device 'cuda:0' \
# --height 576 --width 1024 \
# --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth