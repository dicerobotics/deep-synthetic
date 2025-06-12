Authored by by Arshad MA

# Enhancing Realism of Synthetic MWIR Images with Semantically Guided GANs: A CUT and Grad-CAM Feedback Framework

## Overview

This project addresses the challenge of improving the **realism and semantic fidelity** of synthetic **Mid-Wave Infrared (MWIR)** images, which are essential in defense, surveillance, and night-time imaging applications. Real MWIR data collection is expensive and difficult due to hardware cost, safety concerns, and environmental constraints. As an alternative, simulator-generated images (e.g., from OktalSE) can be used, but they often lack the visual quality and semantic detail required for training high-performing models.

We propose a **three-stage GAN-based framework** combining **Contrastive Unpaired Translation (CUT)**, **Pix2Pix**, and **Grad-CAM feedback** to bridge this simulation-to-reality gap.

---

## Pipeline Overview

### 1. **Pseudo-Pair Generation with CUT**

- Real MWIR images from the **DSIAC ATR dataset** (9,200 images) are mapped to the simulator style using CUT.
- Target domain consists of **300 OktalSE-simulated MWIR images** across varying times of day, seasons, and atmospheric conditions.
- Output: Visually simulated versions of real MWIR images, preserving semantics.

### 2. **Pix2Pix Translation**

- A supervised **Pix2Pix** model learns to translate simulator images into realistic MWIR using CUT-generated pseudo-pairs.
- Training loss:
  - **Adversarial Loss**
  - **L1 Reconstruction Loss**
- This stage improves both perceptual realism and content retention.

### 3. **Semantic Feedback with Grad-CAM**

- A **ResNet18 classifier**, pretrained on MWIR data, is used to generate Grad-CAM attention maps.
- These maps are integrated into Pix2Pix training as a semantic loss term.
- The model is fine-tuned for **200 additional epochs** (201–400), encouraging attention to semantically relevant regions.

---

## Training Details

### Stage 1: CUT Training
- **Source**: Real MWIR (DSIAC ATR)
- **Target**: OktalSE simulated images
- Output: Pseudo-paired dataset

### Stage 2: Pix2Pix Training
- **Initial Training**: 200 epochs with batch size = 1, crop size = 256
- **Losses**: GAN loss + L1 loss
- Optimizer: Adam (lr = 0.0002, β1 = 0.5)

### Stage 3: Semantic Feedback Fine-Tuning
- **Model**: `pix2pix_gradcam`
- **Additional Training**: 200 epochs (epochs 201–400)
- **Losses**: GAN + L1 + Grad-CAM (λ = 10.0 for Grad-CAM loss)

---

## Evaluation Metrics

We used both **perceptual** and **semantic** evaluation metrics:

- **FID**: Fréchet Inception Distance
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **GCSS**: Grad-CAM Structural Similarity
- **CC**: Pearson Correlation Coefficient between attention maps

---

## Results

### Qualitative
- Realistic textures and thermal structures were synthesized.
- Grad-CAM maps showed stronger alignment with semantically relevant regions post-feedback.

### Quantitative
- Substantial reduction in FID and improvements in PSNR and SSIM.
- GCSS and CC validated stronger semantic fidelity.

### Ablation Study
- Models with Grad-CAM feedback consistently outperformed both CUT-only and standard Pix2Pix models.

---

## Conclusion

This work presents a robust framework for enhancing the **realism and semantic quality** of synthetic MWIR images using GANs and classifier feedback. The method enables more reliable synthetic datasets that can significantly aid model training in resource-scarce thermal domains.

---

## Citation

If you use this work, please cite:

@article{arshad2025paper,
title={Enhancing Realism of Synthetic MWIR Images with Semantically Guided GANs: A CUT and Grad-CAM Feedback Framework},
author={Muhammad Awais Arshad, Hyochoong Bang, ...},
journal={Machine Vision and Applications},
year={2025}
}

---

## Contact

For questions or collaborations, please contact:  
📧 m.awais@kaist.ac.kr  
🔬 [Mechanical Engineering Research Institute, KAIST]




# Train and test the CUT Model
Objective: Transferring style from OktalSE simulated images onto MWIR real images and vice versa.

``` bash
cd ./scripts/
./run_real2cut.sh
./run_sim2cut.sh
```

**Output**  
`/workspace/deep-synthetic/results/mwir_real2cut/test_latest/images/ `   
`/workspace/deep-synthetic/results/mwir_sim2cut/test_latest/images/ `   

### **Nomenclature**
**Experiments**  
`mwir_real2cut`: Transferring style from OktalSE simulated images onto MWIR real images.  
`mwir_sim2cut`: Transferring style from MWIR real images onto OktalSE simulated images.  

**Images**  
`mwir_real` : Real MWIR Images from the SENSIAC Dataset placed in datasets/real_A for training.  
`mwir_sim`  : Simulated MWIR image generated with OktalSE simulator placed in datasets/real_B for training.  
`mwir_real2cut` : Images produced by CUT by transferring style of mwir_sim onto mwir_real.  
`mwir_sim2cut`  : Images produced by CUT by transferring style of mwir_real onto mwir_sim.  

### **Directory contents**  
**Experiment:** mwir_real2cut   
`datasets/mwir_real2cut/train_A`: mwir_real    
`datasets/mwir_real2cut/train_B`: mwir_sim    
`datasets/mwir_real2cut/test_A`: mwir_real (unscene during training)  
`datasets/mwir_real2cut/test_B`: mwir_sim  (unscene during training)  
`results/mwir_real2cut/test_lastest/images/real_A/`:   
    - Contains the first `n` images from testA/ — these is your actual input to the CUT model (e.g. mwir_real). `n` defaults to 50  
`results/mwir_real2cut/test_lastest/images/fake_B/`:  
    - Contains the generated outputs from those `n` real_A images. `n` defaults to 50  
`results/mwir_real2cut/test_lastest/images/real_B/`:  
    - Contains `n` random images from testB/, not trainB/, and not aligned with real_A. `n` defaults to 50  

| Folder   | Source Directory      | Purpose                                                 |
| -------- | --------------------- | ------------------------------------------------------- |
| `real_A` | `datasets/.../testA/` | Input images to the model (mwir_real)                   |
| `fake_B` | Generated by model    | Style-transferred with CUT                              |
| `real_B` | `datasets/.../testB/` | Random reference images from target domain B (mwir_sim) |


**Experiment:** mwir_sim2cut  
`datasets/mwir_sim2cut/train_A`: mwir_sim  
`datasets/mwir_sim2cut/train_B`: mwir_real  
`datasets/mwir_sim2cut/test_A`: mwir_sim (unscene during training)   
`datasets/mwir_sim2cut/test_B`: mwir_real (unscene during training)  
`results/mwir_sim2cut/test_lastest/images/real_A/`:  
    - Contains the first `n` images from testA/ — this is your actual input to the CUT model (e.g. mwir_sim). `n` defaults to 50  
`results/mwir_sim2cut/test_lastest/images/fake_B/`:  
    - Contains the generated outputs from those `n` real_A images. `n` defaults to 50  
`results/mwir_sim2cut/test_lastest/images/real_B/`:  
    - Contains `n` random images from testB/, not trainB/, and not aligned with real_A. `n` defaults to 50  

| Folder   | Source Directory      | Purpose                                                  |
| -------- | --------------------- | -------------------------------------------------------- |
| `real_A` | `datasets/.../testA/` | Input images to the model (mwir_sim)                     |
| `fake_B` | Generated by model    | Style-transferred with CUT                               |
| `real_B` | `datasets/.../testB/` | Random reference images from target domain B (mwir_real) |


**Important Notes**:   
CUT is designed for unpaired image-to-image translation.  
The real_B/ folder in the results is just for qualitative comparison — it does not correspond to any particular real_A image unless you're using a paired dataset (which CUT doesn't require).  


### **Image similarity metrices with good result ranges**
| Metric                                                | Based On                       | Measures                       | Robust To                              | Best For               | **Typical Range**              | **Good Result**                                                     |
| ----------------------------------------------------- | ------------------------------ | ------------------------------ | -------------------------------------- | ---------------------- | ------------------------------ | ------------------------------------------------------------------- |
| **PSNR** (Peak Signal-to-Noise Ratio)                 | Pixel-wise error (log scale)   | Pixel-level fidelity           | None (sensitive to small shifts/noise) | Compression, denoising | **\[20 – 50] dB**              | **>30 dB** (acceptable), **>40 dB** (high quality)                  |
| **LPIPS** (Learned Perceptual Image Patch Similarity) | Deep network features          | Perceptual/semantic similarity | Misalignment, texture, color shifts    | GANs, synthesis        | **\[0 – 1]** (lower is better) | **<0.3** (good), **<0.2** (very close), **<0.1** (almost identical) |
| **SSIM** (Structural Similarity Index)                | Luminance, contrast, structure | Structural similarity          | Noise, brightness shifts               | General image quality  | **\[0 – 1]**                   | **>0.85** (good), **>0.95** (very good)                             |


**Our Results**  
Experiment: `mwir_real2cut`  
Samples: 50 (Test set)  

| Comparison              | Value     |
| ----------------------- | --------- |
|PSNR (fake_B vs real_B)  | 16.23 dB  |
|LPIPS (fake_B vs real_B) | 0.3723    | 
|SSIM (fake_B vs real_B)  | 0.6455    |  
| SSIM (real_A vs fake_B) | 0.7259    |  

Samples: 9200 (Complete dataset: train set and test set) 

| Comparison              | Value   |
| ----------------------- | -----   |
| PSNR (fake_B vs real_B) | 13.91 dB|  
| LPIPS (fake_B vs real_B)| 0.3865  |
| SSIM (fake_B vs real_B) | 0.5578  |
| SSIM (real_A vs fake_B) | 0.6943  |


# How to integrate Grad-CAM in Pix2Pix training?
### Basic workflow

**1. Pix2Pix generator**:
Input: source image (e.g., segmentation mask)
Output: generated image (fake photo)

**2. Discriminator**:
Tries to distinguish generated vs real images conditioned on the input

**3. Classifier + Grad-CAM**:
Pretrained classifier frozen (e.g., ResNet trained on real photos)
For each generated image, compute Grad-CAM map for the predicted or target class.

**4. Grad-CAM loss**:
Compare Grad-CAM maps of generated images to real target images’ Grad-CAM maps (or use a target pattern)
This ensures generated images activate the classifier similarly to real images.

**5. Loss to minimize for generator**:

- GAN adversarial loss (Pix2Pix original)

- Reconstruction loss (e.g., L1 between generated & target image)

- Grad-CAM loss (new) to align important regions
