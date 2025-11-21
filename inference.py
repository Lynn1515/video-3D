import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from viewcrafter_vggt import ViewCrafter

from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime

#os.environ["HF_HOME"] = "/mnt/data/hf_cache"


if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    os.makedirs(opts.save_dir,exist_ok=True)
    pvd = ViewCrafter(opts)
    print(f"Running On: {opts.mode} mode")
    if opts.mode == 'single_view_target':
        pvd.nvs_single_view()

    elif opts.mode == 'single_view_nbv':
        pvd.nvs_single_view()#nvs_single_view_w_ctrl()#nvs_two_view()#

    elif opts.mode == 'single_view_eval':
        pvd.nvs_single_view_eval()

    elif opts.mode == 'sparse_view_interp':
        pvd.nvs_sparse_view_interp()

    elif opts.mode == 'single_view_ref_iterative':
        pvd.nvs_single_view_ref_iterative()

    elif opts.mode == 'view_select_iterative':
        pvd.single_view_best_select_iterative()

    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")
