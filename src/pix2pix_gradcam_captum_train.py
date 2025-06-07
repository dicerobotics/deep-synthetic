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
import random
import numpy as np

import torch.nn.functional as F
from checkpoints.classifiers.resnet18_mwir import get_trained_resnet18


# -------Add Reproducibility------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------- Config -------
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
DATA_DIR = "datasets/pix2pix"
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

# Use Captum's Grad-CAM
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

        # ---- Grad-CAM Loss with Captum ----
        gradcam_loss = 0
        for i in range(fake.size(0)):
            # Predict class index from real image
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

    # ---- Save Output Image ----
    save_image(fake * 0.5 + 0.5, f"{SAVE_DIR}/fake_epoch_{epoch+1}.png")

    # ---- Log key metrics per epoch ----
    print(f"[Epoch {epoch+1}] Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, GradCAM: {gradcam_loss.item():.4f}")


# ------- Save Models -------
torch.save(generator.state_dict(), "checkpoints/generator_pix2pix_gradcam.pth")
torch.save(discriminator.state_dict(), "checkpoints/discriminator_pix2pix_gradcam.pth")
