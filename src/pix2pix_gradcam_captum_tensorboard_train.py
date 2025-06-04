"""
Pix2Pix Training Script with Grad-CAM Feedback and TensorBoard Logging
======================================================================

This script implements a modified Pix2Pix GAN architecture designed to enhance 
the realism of simulator-generated images by incorporating semantic feedback from a 
pretrained ResNet18 classifier using Grad-CAM (via Captum). It integrates:

- A U-Net based generator and a patch-based discriminator
- Grad-CAM similarity loss between real and generated images to preserve salient features
- TensorBoard logging for real-time monitoring
- Image and Grad-CAM heatmap saving for qualitative analysis

Intended Use:
--------------
Use this script to train on paired datasets (simulated → real) where structural and 
semantic realism is crucial, e.g., MWIR military datasets. Output images, heatmaps, 
and training metrics are stored under `outputs/` and `runs/`.

Dependencies:
--------------
- PyTorch, torchvision
- Captum (for Grad-CAM)
- TensorBoard
- PIL, matplotlib, tqdm

Author: [Arshad MA]
"""



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from captum.attr import LayerGradCam
import torch.nn.functional as F
from resnet18_mwir import get_trained_resnet18  # Load your trained classifier
import matplotlib.pyplot as plt

# Optional: TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/pix2pix_gradcam")

# ------- Config -------
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
DATA_DIR = "./datasets/pix2pix"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "tmp"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ------- Dataset -------
class PairedDataset(Dataset):
    def __init__(self, root, transform=None):
        self.sim_dir = os.path.join(root, "sim")
        self.real_dir = os.path.join(root, "real")
        self.transform = transform
        self.filenames = sorted(os.listdir(self.sim_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sim_path = os.path.join(self.sim_dir, self.filenames[idx])
        real_path = os.path.join(self.real_dir, self.filenames[idx])
        sim_img = Image.open(sim_path).convert("RGB")
        real_img = Image.open(real_path).convert("RGB")

        if self.transform:
            sim_img = self.transform(sim_img)
            real_img = self.transform(real_img)

        return sim_img, real_img

# ------- Transform -------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------- Pix2Pix Models -------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

# ------- Instantiate -------
generator = UNetGenerator().to(DEVICE)
discriminator = PatchDiscriminator().to(DEVICE)
classifier = get_trained_resnet18().to(DEVICE)
classifier.eval()

target_layer = classifier.layer4[1]  # Modify index if needed
gradcam = LayerGradCam(classifier, target_layer)

# ------- Optimizers and Losses -------
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# ------- Dataloader -------
dataset = PairedDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------- Training Loop -------
for epoch in range(NUM_EPOCHS):
    for sim, real in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        sim, real = sim.to(DEVICE), real.to(DEVICE)

        # ---- Train Discriminator ----
        fake = generator(sim)
        real_pair = torch.cat((sim, real), 1)
        fake_pair = torch.cat((sim, fake.detach()), 1)

        pred_real = discriminator(real_pair)
        pred_fake = discriminator(fake_pair)

        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ---- Train Generator ----
        fake_pair = torch.cat((sim, fake), 1)
        pred_fake = discriminator(fake_pair)
        loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = criterion_L1(fake, real) * 100

        gradcam_loss = 0
        for i in range(fake.size(0)):
            with torch.no_grad():
                logits = classifier(real[i].unsqueeze(0))
                target_class = logits.argmax(dim=1).item()

            heatmap_fake = gradcam.attribute(fake[i].unsqueeze(0), target=target_class)
            heatmap_real = gradcam.attribute(real[i].unsqueeze(0), target=target_class)

            heatmap_fake = F.interpolate(heatmap_fake, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')
            heatmap_real = F.interpolate(heatmap_real, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')

            gradcam_loss += criterion_L1(heatmap_fake, heatmap_real)

        gradcam_loss = gradcam_loss / fake.size(0)

        loss_G = loss_G_GAN + loss_G_L1 + gradcam_loss

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    # ---- Logging ----
    print(f"[Epoch {epoch+1}] Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | L1: {loss_G_L1.item():.4f} | GradCAM: {gradcam_loss.item():.4f}")
    with open("loss_log.csv", "a") as f:
        f.write(f"{epoch+1},{loss_D.item():.4f},{loss_G.item():.4f},{loss_G_L1.item():.4f},{gradcam_loss.item():.4f}\n")

    # ---- TensorBoard Logging ----
    writer.add_scalar("Loss/Discriminator", loss_D.item(), epoch)
    writer.add_scalar("Loss/Generator", loss_G.item(), epoch)
    writer.add_scalar("Loss/GradCAM", gradcam_loss.item(), epoch)
    writer.add_images("Fake", fake * 0.5 + 0.5, epoch)
    writer.add_images("Real", real * 0.5 + 0.5, epoch)
    writer.add_images("Sim", sim * 0.5 + 0.5, epoch)

    # ---- Save Output Image ----
    save_image(fake * 0.5 + 0.5, f"{SAVE_DIR}/fake_epoch_{epoch+1}.png")
    save_image(sim * 0.5 + 0.5, f"{SAVE_DIR}/sim_epoch_{epoch+1}.png")
    save_image(real * 0.5 + 0.5, f"{SAVE_DIR}/real_epoch_{epoch+1}.png")

    # ---- Save Heatmaps ----
    def save_heatmap(tensor, name):
        tensor = tensor.squeeze().cpu().detach().numpy()
        plt.imshow(tensor, cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
        plt.close()

    save_heatmap(heatmap_fake[0], f"{SAVE_DIR}/heatmap_fake_epoch_{epoch+1}.png")
    save_heatmap(heatmap_real[0], f"{SAVE_DIR}/heatmap_real_epoch_{epoch+1}.png")

# ------- Save Models -------
torch.save(generator.state_dict(), "checkpoints/generator_pix2pix_gradcam.pth")
torch.save(discriminator.state_dict(), "checkpoints/discriminator_pix2pix_gradcam.pth")
