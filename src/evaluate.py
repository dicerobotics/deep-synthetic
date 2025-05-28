
import os
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from skimage.metrics import structural_similarity as ssim_sk
import numpy as np
import lpips
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import rgb_to_grayscale




# ----- Setup -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_model = lpips.LPIPS(net='alex').to(device)

to_tensor = T.Compose([
    T.Resize((256, 256)),   # Resize if needed
    T.Grayscale(num_output_channels=1),  # Ensure single channel
    T.ToTensor()
])

to_tensor_rgb = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# ----- Metrics -----
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(img1, img2):
    # Resize
    if img1.shape[1:] != img2.shape[1:]:
        img2 = TF.resize(img2, size=img1.shape[1:])

    # Match channels
    if img1.shape[0] != img2.shape[0]:
        if img1.shape[0] == 1 and img2.shape[0] == 3:
            img2 = rgb_to_grayscale(img2, num_output_channels=1)
        elif img1.shape[0] == 3 and img2.shape[0] == 1:
            img2 = img2.expand(3, -1, -1)

    # Convert to numpy
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()

    # print(f"[DEBUG] SSIM input shapes: img1 {img1_np.shape}, img2 {img2_np.shape}")
    return ssim_sk(img1_np, img2_np, data_range=1.0, channel_axis=-1)



def compute_lpips(img1, img2):
    img1 = img1.expand(1, 3, -1, -1)  # Ensure 3 channels
    img2 = img2.expand(1, 3, -1, -1)
    return lpips_model(img1.to(device), img2.to(device)).item()

# ----- Evaluation -----
def evaluate_dirs(real_A_dir, fake_B_dir, real_B_dir):
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    files = sorted(os.listdir(fake_B_dir))

    for fname in files:
        path_fake = os.path.join(fake_B_dir, fname)
        path_real_A = os.path.join(real_A_dir, fname)
        path_real_B = os.path.join(real_B_dir, fname)

        if not (os.path.exists(path_real_A) and os.path.exists(path_real_B)):
            continue

        # Load images
        fake = to_tensor_rgb(Image.open(path_fake))
        real_A = to_tensor(Image.open(path_real_A))
        real_B = to_tensor_rgb(Image.open(path_real_B))


        # Ensure float tensors in [0,1]
        fake = fake.to(torch.float32)
        real_A = real_A.to(torch.float32)
        real_B = real_B.to(torch.float32)

        # Metric calculations
        psnr_vals.append(compute_psnr(fake, real_B))
        lpips_vals.append(compute_lpips(fake, real_B))
        print(f"[INFO] real_A shape: {real_A.shape}, fake shape: {fake.shape}")

        ssim_vals.append(compute_ssim(fake, real_B))


    print(f"Samples compared: {len(psnr_vals)}")
    print(f"→ PSNR (fake_B vs real_B): {np.mean(psnr_vals):.2f} dB")
    print(f"→ LPIPS (fake_B vs real_B): {np.mean(lpips_vals):.4f}")
    print(f"→ SSIM (fake_B vs real_B): {np.mean(ssim_vals):.4f}")
    
# ----- Run -----
# Evaluate CUT results
evaluate_dirs(
    real_A_dir='/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/images/real_A',
    fake_B_dir='/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/images/fake_B',
    real_B_dir='/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/images/real_B'
)