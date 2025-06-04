import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# ---- Dataset ----
class PairedDataset(Dataset):
    def __init__(self, sim_dir, real_dir, transform):
        self.sim_images = sorted(os.listdir(sim_dir))
        self.sim_dir = sim_dir
        self.real_dir = real_dir
        self.transform = transform

    def __len__(self):
        return len(self.sim_images)

    def __getitem__(self, idx):
        sim_path = os.path.join(self.sim_dir, self.sim_images[idx])
        real_path = os.path.join(self.real_dir, self.sim_images[idx])
        sim = self.transform(Image.open(sim_path).convert("RGB"))
        real = self.transform(Image.open(real_path).convert("RGB"))
        return sim, real

# ---- Generator & Discriminator (Pix2Pix) ----
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal UNet; use torchvision implementation or full Pix2Pix UNet for better results
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# ---- Training Configuration ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_data = PairedDataset("../datasets/pix2pix/train/sim", "../datasets/pix2pix/train/real", transform)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# ---- Training Loop ----
EPOCHS = 50
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for sim, real in loop:
        sim, real = sim.to(device), real.to(device)

        # Train Discriminator
        fake = G(sim).detach()
        pred_real = D(sim, real)
        pred_fake = D(sim, fake)
        loss_D = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                        criterion_GAN(pred_fake, torch.zeros_like(pred_fake)))
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        fake = G(sim)
        pred_fake = D(sim, fake)
        loss_G = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) + \
                 100 * criterion_L1(fake, real)  # λ=100
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

    # Save samples
    if (epoch + 1) % 10 == 0:
        save_image(fake * 0.5 + 0.5, f"../tmp/fake_epoch_{epoch+1}.png")
        save_image(real * 0.5 + 0.5, f"../tmp/real_epoch_{epoch+1}.png")

# Save models
torch.save(G.state_dict(), "../models/pix2pix/generator_weights.pth")
torch.save(D.state_dict(), "../models/pix2pix/discriminator_weights.pth")
