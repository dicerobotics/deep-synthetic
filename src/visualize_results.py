import os
import csv
import torch
import lpips
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToTensor, Resize
from torchvision.utils import make_grid
from math import log10

# --- Helper Functions ---

def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    grid = make_grid(tensor, nrow=1)
    ndarr = grid.mul(255).byte().squeeze().permute(1, 2, 0).numpy()
    return Image.fromarray(ndarr)

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    return 20 * log10(1.0 / (mse ** 0.5 + 1e-10))

def compute_ssim(img1, img2):
    # img1 = TF.rgb_to_grayscale(img1).squeeze().numpy()
    # img2 = TF.rgb_to_grayscale(img2).squeeze().numpy()
    img1 = TF.rgb_to_grayscale(img1).squeeze().cpu().numpy()
    img2 = TF.rgb_to_grayscale(img2).squeeze().cpu().numpy()
    return ssim(img1, img2, data_range=1.0)

def save_grid(real_A, fake_B, real_B, out_path, idx):
    images = torch.stack([real_A, fake_B, real_B])
    grid = make_grid(images, nrow=3, padding=5)
    npimg = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(12, 4))
    plt.imshow(npimg)
    plt.axis('off')
    plt.title("real_A | fake_B | real_B")
    plt.tight_layout()
    plt.savefig(f"{out_path}/sample_{idx:03d}.png")
    plt.close()

# --- Main Evaluation Logic ---

def evaluate_and_visualize(real_A_dir, fake_B_dir, real_B_dir, out_dir, output_csv):
    os.makedirs(out_dir, exist_ok=True)

    loss_fn_vgg = lpips.LPIPS(net='alex').cuda().eval()

    image_files = sorted(os.listdir(real_A_dir))
    csv_rows = [("image", "PSNR", "SSIM", "LPIPS")]

    for i, filename in enumerate(image_files):
        path_real_A = os.path.join(real_A_dir, filename)
        path_fake_B = os.path.join(fake_B_dir, filename)
        path_real_B = os.path.join(real_B_dir, filename)

        # Load and preprocess images
        pil_real_A = Image.open(path_real_A).convert("RGB").resize((256, 256))
        pil_fake_B = Image.open(path_fake_B).convert("RGB").resize((256, 256))
        pil_real_B = Image.open(path_real_B).convert("RGB").resize((256, 256))

        real_A = ToTensor()(pil_real_A).cuda().unsqueeze(0)
        fake_B = ToTensor()(pil_fake_B).cuda().unsqueeze(0)
        real_B = ToTensor()(pil_real_B).cuda().unsqueeze(0)

        # Save grid
        save_grid(real_A.squeeze(0), fake_B.squeeze(0), real_B.squeeze(0), out_dir, i)

        # Metrics
        psnr_val = compute_psnr(fake_B, real_B)
        ssim_val = compute_ssim(real_A.squeeze(0), fake_B.squeeze(0))
        lpips_val = loss_fn_vgg(fake_B, real_B).item()

        csv_rows.append((filename, psnr_val, ssim_val, lpips_val))

    # Save CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"✔ Visualization saved to: {out_dir}")
    print(f"✔ Metrics CSV saved to: {output_csv}")


evaluate_and_visualize(
    real_A_dir='/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/images/real_A',
    fake_B_dir='/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/images/fake_B',
    real_B_dir='/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/images/real_B'
    out_dir="/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/visuals",
    output_csv="/workspace/deep-synthetic/third_party/_CUT/results/mwir2sim_cut/test_latest/eval_metrics.csv"
)