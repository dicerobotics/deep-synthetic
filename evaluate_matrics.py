import os
import sys
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as compare_ssim


# Data Directories

# REAL_MWIR = './datasets/mwir_pseudopaired_cut20/real_A/test'
REAL_MWIR = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/real_B'
REAL_MWIR_HEATMAP = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/gradcam_real_B'

OKTAL_SE_MWIR = './results/oktal_target/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/real_B'
OKTAL_SE_MWIR_HEATMAP = './results/oktal_target/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/gradcam_real_B'

CUT_TRANSLATED_REAL_MWIR = './results/psim_target/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/real_B'
CUT_TRANSLATED_REAL_MWIR_HEATMAP = './results/psim_target/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/gradcam_real_B'

CUT_TRANSLATED_MWIR_TO_PIX2PIX = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_0.0001/test_latest/images/fake_B'
CUT_TRANSLATED_MWIR_TO_PIX2PIX_HEATMAP = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_0.0001/test_latest/images/gradcam_fake_B'

CUT_TRANSLATED_MWIR_TO_PIX2PIXGRADCAM = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/fake_B'
CUT_TRANSLATED_MWIR_TO_PIX2PIXGRADCAM_HEATMAP = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/gradcam_fake_B'

OKTAL_SE_MWIR_TO_PIX2PIX = './results/oktal_start/pix2pix_gradcam_psim2preal_1_10_0.0001/test_latest/images/fake_B'
OKTAL_SE_MWIR_TO_PIX2PIX_HEATMAP = './results/oktal_start/pix2pix_gradcam_psim2preal_1_10_0.0001/test_latest/images/gradcam_fake_B'

OKTAL_SE_MWIR_TO_PIX2PIX_GRADCAM = './results/oktal_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/fake_B'
OKTAL_SE_MWIR_TO_PIX2PIX_GRADCAM_HEATMAP = './results/oktal_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/gradcam_fake_B'

real_dir = REAL_MWIR
real_cam_dir = REAL_MWIR_HEATMAP


# --- Config Evaluation Scenarios---
# Scenario      Description
# 1             Simulator (Oktal-SE) Output 
# 2             Simulator (Oktal-SE) → Pix2Pix 
# 3             Simulator (Oktal-SE) → Pix2Pix-GradCAM 
# 4             Contrastive Unpaired Translation (CUT) Pseudopairs 
# 5             CUT Pseudopairs → Pix2Pix 
# 6             CUT Pseudopairs → Pix2Pix-GradCAM 

evaluation_scenario = 4

match evaluation_scenario:
    case 1: # "Simulator (Oktal-SE) Output":
        print("Evaluating matrics for Simulator (Oktal-SE) Output")
        target_dir = OKTAL_SE_MWIR
        target_cam_dir = OKTAL_SE_MWIR_HEATMAP
    case 2: #"Simulator (Oktal-SE) → Pix2Pix"
        print("Evaluating matrics for Simulator (Oktal-SE) → Pix2Pix")
        target_dir = OKTAL_SE_MWIR_TO_PIX2PIX
        target_cam_dir = OKTAL_SE_MWIR_TO_PIX2PIX_HEATMAP
    case 3: #"Simulator (Oktal-SE) → Pix2Pix-GradCAM"
        print("Evaluating matrics for Simulator (Oktal-SE) → Pix2Pix-GradCAM")
        target_dir = OKTAL_SE_MWIR_TO_PIX2PIX_GRADCAM
        target_cam_dir = OKTAL_SE_MWIR_TO_PIX2PIX_GRADCAM_HEATMAP
    case 4: #"Contrastive Unpaired Translation (CUT) Pseudopairs"
        print("Evaluating matrics for Contrastive Unpaired Translation (CUT) Pseudopairs")
        target_dir = CUT_TRANSLATED_REAL_MWIR
        target_cam_dir = CUT_TRANSLATED_REAL_MWIR_HEATMAP
    case 5: #"CUT Pseudopairs → Pix2Pix"
        print("Evaluating matrics for CUT Pseudopairs → Pix2Pix")
        target_dir = CUT_TRANSLATED_MWIR_TO_PIX2PIX
        target_cam_dir = CUT_TRANSLATED_MWIR_TO_PIX2PIX_HEATMAP
    case 6: #"CUT Pseudopairs → Pix2Pix-GradCAM"
        print("Evaluating matrics for CUT Pseudopairs → Pix2Pix-GradCAM")
        target_dir = CUT_TRANSLATED_MWIR_TO_PIX2PIXGRADCAM
        target_cam_dir = CUT_TRANSLATED_MWIR_TO_PIX2PIXGRADCAM_HEATMAP
    case _:
        sys.exit("Error: No valid matching case for evaluation scenario. Aborting operations.")



# real_dir = REAL_MWIR
# target_dir = CUT_TRANS_TO_PIX2PIX #CUT_TRANS_TO_PIX2PIXGRADCAM #OKTAL_SE_MWIR #CUT_TRANS_MWIR
# real_cam_dir = REAL_MWIR_HEATMAP      # Optional if you have Grad-CAM heatmaps saved as images
# target_cam_dir = CUT_TRANS_TO_PIX2PIX_HEATMAP #CUT_TRANS_TO_PIX2PIXGRADCAM_HEATMAP      # Optional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper functions ---

def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img

def load_grayscale_image(path, size=None):
    img = Image.open(path).convert('L')
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img

def pil_to_tensor(img):
    # Converts PIL image to tensor normalized to [0,1]
    transform = transforms.ToTensor()
    return transform(img)

def load_images_from_folder(folder, size=None):
    paths = sorted(glob.glob(os.path.join(folder, '*')))
    images = [pil_to_tensor(load_image(p, size)) for p in paths]
    return images, paths

def load_heatmaps_from_folder(folder, size=None):
    paths = sorted(glob.glob(os.path.join(folder, '*')))
    heatmaps = [pil_to_tensor(load_grayscale_image(p, size)) for p in paths]
    return heatmaps, paths

# --- Load images ---
print("Loading real images...")
real_images, real_paths = load_images_from_folder(real_dir)
print("Loading generated images...")
fake_images, fake_paths = load_images_from_folder(target_dir)

assert len(real_images) == len(fake_images), "Mismatch in number of real and generated images"

# If you have Grad-CAM heatmaps saved, load them here:
if os.path.exists(real_cam_dir) and os.path.exists(target_cam_dir):
    print("Loading Grad-CAM heatmaps...")
    real_cams, _ = load_heatmaps_from_folder(real_cam_dir)
    fake_cams, _ = load_heatmaps_from_folder(target_cam_dir)
    assert len(real_cams) == len(fake_cams) == len(real_images), "Heatmaps count mismatch"
else:
    real_cams = None
    fake_cams = None

# --- Metrics calculation functions ---

def compute_psnr(real_imgs, fake_imgs):
    psnr_vals = []
    for real, fake in zip(real_imgs, fake_imgs):
        real_np = (real.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        fake_np = (fake.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        psnr = compare_psnr(real_np, fake_np, data_range=255)
        psnr_vals.append(psnr)
    return np.mean(psnr_vals)

def compute_correlation_coefficient(cam_real, cam_fake):
    cc_vals = []
    for real, fake in zip(cam_real, cam_fake):
        real_np = real.numpy().flatten()
        fake_np = fake.numpy().flatten()
        if np.std(real_np) == 0 or np.std(fake_np) == 0:
            cc_vals.append(0)
        else:
            corr, _ = pearsonr(real_np, fake_np)
            cc_vals.append(corr)
    return np.mean(cc_vals)

def compute_gcss(cam_real, cam_fake):
    gcss_vals = []
    for real, fake in zip(cam_real, cam_fake):
        real_np = real.numpy().flatten()
        fake_np = fake.numpy().flatten()
        real_norm = real_np / (np.linalg.norm(real_np) + 1e-8)
        fake_norm = fake_np / (np.linalg.norm(fake_np) + 1e-8)
        cos_sim = np.dot(real_norm, fake_norm)
        gcss_vals.append(cos_sim)
    return np.mean(gcss_vals)


def compute_ssim(cam_real, cam_fake):
    """
    Compute the average SSIM between real and fake CAM images.
    Assumes inputs are torch tensors with shape (1, H, W) or (H, W).
    """
    ssim_vals = []
    
    for real, fake in zip(cam_real, cam_fake):
        # Convert to numpy arrays and ensure 2D shape
        real_np = real.squeeze().detach().cpu().numpy()
        fake_np = fake.squeeze().detach().cpu().numpy()
        
        # Ensure the arrays are float64 for skimage and normalized to [0,1]
        real_np = real_np.astype(np.float64)
        fake_np = fake_np.astype(np.float64)
        
        # Calculate SSIM with specified data_range
        data_range = real_np.max() - real_np.min()
        if data_range == 0:
            ssim_vals.append(1.0 if np.allclose(real_np, fake_np) else 0.0)
        else:
            ssim, _ = compare_ssim(real_np, fake_np, full=True, data_range=data_range)
            ssim_vals.append(ssim)

    return float(np.mean(ssim_vals))


# --- FID calculation ---

print("Calculating FID score...")
fid_value = calculate_fid_given_paths([real_dir, target_dir], batch_size=50, device=device, dims=2048)

# --- PSNR calculation ---
print("Calculating PSNR...")
psnr_value = compute_psnr(real_images, fake_images)

# --- GCSS and CC calculation ---
if real_cams is not None and fake_cams is not None:
    print("Computing GCSS...")
    gcss_value = compute_gcss(real_cams, fake_cams)
    
    print("Computing Correlation Coefficient...")
    cc_value = compute_correlation_coefficient(real_cams, fake_cams)
    
    print("Computing SSIM...")
    ssim_value = compute_ssim(real_cams, fake_cams)
else:
    gcss_value = None
    cc_value = None
    ssim_value = None


# --- Print results ---

print(f"FID Score: {fid_value:.4f}")
print(f"PSNR: {psnr_value:.4f} dB")
if gcss_value is not None and cc_value is not None:
    print(f"GCSS: {gcss_value:.4f}")
    print(f"Correlation Coefficient (CC): {cc_value:.4f}")
    print(f"CAM SSIM: {ssim_value:.4f}")

else:
    print("Grad-CAM heatmaps not provided, skipping GCSS and CC.")


"""
How to use:
Replace the paths in real_dir, fake_dir, real_cam_dir, and fake_cam_dir with your actual folders.

If you do not have Grad-CAM heatmaps saved as images, skip those parts and GCSS, CC, and SSIM will be skipped.

Make sure your images and heatmaps are aligned (same count and order).
"""


# After printing results

with open("evaluation_results.txt", "w") as f:
    f.write(f"FID Score: {fid_value:.4f}\n")
    f.write(f"PSNR: {psnr_value:.4f} dB\n")
    if gcss_value is not None and cc_value is not None:
        f.write(f"GCSS: {gcss_value:.4f}\n")
        f.write(f"Correlation Coefficient (CC): {cc_value:.4f}\n")
        f.write(f"SSIM: {ssim_value:.4f}\n")
    else:
        f.write("Grad-CAM heatmaps not provided, skipping GCSS, CC, and SSIM.\n")

print("Results saved to evaluation_results.txt")