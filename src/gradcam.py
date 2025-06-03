import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

# 1. Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# 2. Load and preprocess image
# img_path = "./../tmp/cat.jpeg"  # Use your own image path
img_path = "./../tmp/dog.jpg"  # Use your own image path
img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)  # Add batch dim

# 3. Choose the target class (e.g., top-1 prediction)
output = model(input_tensor)
target_class = output.argmax(dim=1).item()

# 4. Define target layer (usually the last conv layer)
target_layer = model.layer4

# 5. Initialize LayerGradCam
gradcam = LayerGradCam(model, target_layer)

# 6. Compute attribution
attributions = gradcam.attribute(input_tensor, target=target_class)

# 7. Upsample to input size
upsampled = LayerAttribution.interpolate(attributions, input_tensor.shape[2:])

# 8. Prepare attribution for visualization
attr = upsampled.squeeze().cpu().detach().numpy()  # shape: (224, 224)
attr = np.expand_dims(attr, axis=2)  # shape: (224, 224, 1)

# --- Save Grad-CAM heatmap alone ---

fig, ax = viz.visualize_image_attr(
    attr,
    method="heat_map",
    sign="positive",
    show_colorbar=True,
    title="Grad-CAM Heatmap",
    use_pyplot=False  # Disable automatic plt.show()
)
fig.savefig("./../tmp/gradcam_output.png")
plt.close(fig)

print("Grad-CAM heatmap saved to gradcam_output.png")

# --- Save Grad-CAM overlay on original image ---

# Convert original PIL image to numpy and normalize between 0 and 1 for overlay
img_np = np.array(img.resize((224, 224))) / 255.0

fig2, ax2 = viz.visualize_image_attr(
    attr,
    original_image=img_np,
    method="blended_heat_map",
    sign="positive",
    show_colorbar=True,
    title="Grad-CAM Overlay",
    use_pyplot=False
)

fig2.savefig("./../tmp/gradcam_overlay.png")
plt.close(fig2)

print("Grad-CAM overlay image saved to gradcam_overlay.png")
