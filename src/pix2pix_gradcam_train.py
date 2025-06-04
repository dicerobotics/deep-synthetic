import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from gradcam import GradCAM  # Assumed to be defined elsewhere

from resnet_mwir import get_trained_resnet18  # Load your trained classifier

# ------- Config -------
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
DATA_DIR = "./datasets/sim2real"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

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
gradcam = GradCAM(model=classifier, target_layer="layer4")

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

        # ---- Grad-CAM loss ----
        gradcam_loss = 0
        for i in range(fake.size(0)):
            heatmap_fake = gradcam(fake[i].unsqueeze(0))
            heatmap_real = gradcam(real[i].unsqueeze(0))
            gradcam_loss += criterion_L1(heatmap_fake, heatmap_real)
        gradcam_loss = gradcam_loss / fake.size(0)

        loss_G = loss_G_GAN + loss_G_L1 + gradcam_loss

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    # ---- Save Output Image ----
    save_image(fake * 0.5 + 0.5, f"{SAVE_DIR}/fake_epoch_{epoch+1}.png")

# ------- Save Models -------
torch.save(generator.state_dict(), "models/generator_pix2pix_gradcam.pth")
torch.save(discriminator.state_dict(), "models/discriminator_pix2pix_gradcam.pth")
