# GSoC 2026 | ML4Sci EXXA 

**Candidate:** Divyansh Soni  
**Target Project:** EXXA2  
**Status:** Completed (General Test + Image-Based Test)

---

## Overview

This repository contains the solutions for the **ML4Sci GSoC 2026** test tasks. The project focuses on applying physics-informed deep learning to analyze synthetic ALMA observations of protoplanetary disks,the birthplaces of planets.

The core contribution is **DiskVAE**, a specialized Variational Autoencoder designed to preserve the faint, fine-grained ring structures of protoplanetary disks that standard computer vision models often blur out. This model is key to the **Image-Based Test** (EXXA2), offering high-fidelity reconstruction and a structured latent space for scientific analysis.

---

## Repository Structure

```bash
gsoc-2026-exxa/
├── continuum_data_subset/            # Dataset (excluded from repo via .gitignore)
├── general_test/
│   ├── General_Test.ipynb            # [TASK 1] Unsupervised clustering pipeline
│   ├── general_test_pipeline_v2.svg  # Methodology diagram
│   └── clustermap.png                # Clustering visualization
│
├── image_task/
│   ├── Image_Test.ipynb              # [TASK 2] DiskVAE Model & Inference Pipeline
│   ├── diskvae_full_architecture3.svg # Model Architecture
│   ├── reconstructed_image.png       # Sample reconstruction
│   ├── resultplots.png               # Training metrics & performance
│   └── task2.pth                     # Pre-trained DiskVAE model weights
│
└── README.md                         # Documentation
```

---

## Task 1: General Test

**Goal:** Unsupervised clustering of disks to identify properties (specifically planets) without labels.

### The Methodology
1.  **Preprocessing:** Arcsinh stretch, normalization, and central star masking.
2.  **Feature Extraction:** Used a custom Convolutional Autoencoder to compress images into a compact latent representation.
3.  **Clustering:** Applied **Gaussian Mixture Models (GMM)** on the latent vectors to group disks by morphology.
4.  **Analysis:** The resulting clusters clearly separated transition disks, multi-ring systems, and smooth disks.

#### Pipeline Diagram
![Task 1 Pipeline](general_test/general_test_pipeline_v2.svg)

### Clustering Results
The clustering successfully separates disks based on their morphological features. Below is the visualization of the learned clusters.
![Clustering Map](general_test/clustermap.png)

### Conclusion
The UMAP projection reveals three distinguishable morphological clusters formed purely from the latent space, with no planet count information used during clustering. Overlaying the detected planet candidates shows that disk morphology and planetary features are not independent , there is a meaningful relationship between the two but morphology alone is not a sufficient predictor of whether a planet is present. The majority of the latent space shows mixing between planet groups.
Two signals do emerge with confidence at the extremes. The top-left cluster is strongly enriched in planet-free disks, while the bottom-right cluster contains predominantly disks hosting one or more planets. Disks that fall in these extreme morphological regions carry a noticeably higher predictive signal than those in the central mixed region, suggesting that certain structural features — localised brightness enhancements, ring asymmetries, or gap depths are genuinely associated with planet formation activity, even if the full relationship cannot be captured by morphology alone.

The notebook `general_test/General_Test.ipynb` contains the full pipeline from raw FITS files to visualized clusters.

---
##  Task 2: Image-Based Test (Primary Focus for EXXA2)

**Goal:** Train an autoencoder to reconstruct protoplanetary disk images with an accessible latent space.

### The Solution: DiskVAE (Ring-Aware VAE)
Standard generic autoencoders struggle with the specific geometry of astronomical disks, often treating rings as noise or blurring them into a smooth gradient. **DiskVAE** introduces geometric priors directly into the architecture:

#### Architecture
![pipelineArchitecture](image_task/diskvae_full_architecture3.svg)

1.  **Radial Conditioning:** Explicitly injects a polar coordinate grid into every layer, grounding the model in the physical reality of the disk's center-out structure.
2.  **Attention-Weighted Loss:** Uses pre-computed "Clean" (structure-only) and "Pointness" (planet-candidate) maps to weight the loss function locally.
    *   **Rings** get **1000x** validation signal.
    *   **Planet candidates** get **5000x** validation signal.
3.  **Joint Planet Head:** A specialized auxiliary head trained simultaneously to detect planet signatures from the latent bottleneck.
4.  **Bottleneck Self-Attention:** Ensures global coherence of rings (symmetry) across the image.
![ModelArchitecture](image_task/diskVAE_task2_archi.svg)
### Performance & Metrics
The model was evaluated on a held-out test set (20% split).

| Metric | Score | Note |
| :--- | :--- | :--- |
| **MSE** | **~0.00003** | Extremely low reconstruction error. |
| **MS-SSIM** | **~0.9989** | High structural similarity (captures rings/gaps). |
| **Latent Dim** | **128** | Dense, accessible latent space for analysis. |

#### Reconstruction Sample
Here is a sample reconstruction from the test set, showing the input, reconstruction, residual (difference), and planet probability map.
![Reconstruction](image_task/reconstructed_image.png)

### Training Results
![Results](image_task/resultplots.png)
### Inference
To run inference on new data:
1.  Open `image_task/Image_Test.ipynb`.
2.  Use the `run_inference(folder_path)` function provided in the notebook.
3.  It returns reconstructions, latent vectors, and planet probability maps.

---
### Running the Notebooks
**Both notebooks are self-contained.**

1.  **Setup Data:** Place the `.fits` files in a folder named `continuum_data_subset/` in the root directory.
2.  **Run Image Task (EXXA2):**
    *   Navigate to `image_task/`.
    *   Run `Image_Test.ipynb`.
    *   Set `TRAIN_FROM_SCRATCH = False` to use the provided `task2.pth` weights.
3.  **Run General Task:**
    *   Navigate to `general_test/`.
    *   Run `General_Test.ipynb`.

