import os
import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from cleanfid import fid
from tqdm import tqdm

def load_image(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def compute_psnr_ssim(real_dir, gen_dir):
    psnr_vals, ssim_vals = [], []
    for fname in tqdm(os.listdir(real_dir)):
        real_img = load_image(os.path.join(real_dir, fname))
        gen_img = load_image(os.path.join(gen_dir, fname))

        psnr_vals.append(psnr(real_img, gen_img, data_range=255))
        ssim_vals.append(ssim(real_img, gen_img, channel_axis=-1))

    return np.mean(psnr_vals), np.mean(ssim_vals)

def compute_lpips(real_dir, gen_dir, net='alex'):
    loss_fn = lpips.LPIPS(net=net)
    lpips_vals = []
    for fname in tqdm(os.listdir(real_dir)):
        real_img = load_image(os.path.join(real_dir, fname)) / 255.0
        gen_img = load_image(os.path.join(gen_dir, fname)) / 255.0

        real_tensor = torch.tensor(real_img).permute(2, 0, 1).unsqueeze(0).float()
        gen_tensor = torch.tensor(gen_img).permute(2, 0, 1).unsqueeze(0).float()
        dist = loss_fn(real_tensor, gen_tensor)
        lpips_vals.append(dist.item())
    return np.mean(lpips_vals)

def compute_gcss_cc(real_dir, gen_dir):
    gcss_vals, cc_vals = [], []
    for fname in tqdm(os.listdir(real_dir)):
        real_img = load_image(os.path.join(real_dir, fname))
        gen_img = load_image(os.path.join(gen_dir, fname))

        # Fake Grad-CAM maps (placeholder) via grayscale for example
        real_map = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
        gen_map = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)

        real_map = real_map / 255.0
        gen_map = gen_map / 255.0

        # GCSS: Structural Similarity between activation maps
        gcss_vals.append(ssim(real_map, gen_map))

        # CC: Correlation Coefficient
        cc_vals.append(pearsonr(real_map.flatten(), gen_map.flatten())[0])
    return np.mean(gcss_vals), np.mean(cc_vals)

def compute_fid_score(real_dir, gen_dir):
    return fid.compute_fid(real_dir, gen_dir)

if __name__ == "__main__":
    real_dir = 'real_dir'  # ground-truth MWIR
    gen_dir = 'gen_dir'    # generated MWIR

    print("Calculating FID...")
    fid_score = compute_fid_score(real_dir, gen_dir)

    print("Calculating PSNR and SSIM...")
    psnr_val, ssim_val = compute_psnr_ssim(real_dir, gen_dir)

    print("Calculating LPIPS...")
    lpips_val = compute_lpips(real_dir, gen_dir)

    print("Calculating GCSS and CC...")
    gcss_val, cc_val = compute_gcss_cc(real_dir, gen_dir)

    print("\n=== Evaluation Results ===")
    print(f"FID   : {fid_score:.2f}")
    print(f"PSNR  : {psnr_val:.2f} dB")
    print(f"SSIM  : {ssim_val:.4f}")
    print(f"LPIPS : {lpips_val:.4f}")
    print(f"GCSS  : {gcss_val:.4f}")
    print(f"CC    : {cc_val:.4f}")
