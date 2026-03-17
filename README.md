# ML4Sci EXXA | GSoC 2026
## Planet-Focused Ring-Aware VAE & Unsupervised Disk Clustering

**Candidate:** Divyansh Soni  
**Project:** Machine Learning for Science (ML4Sci) - EXXA

---

## Description
Protoplanetary disks are the birthplaces of planets, but detecting young planets embedded within them is challenging due to the overwhelming brightness of the host star and the complex ring structures of the disk itself. This project implements two distinct deep learning approaches to tackle this inference challenge:

1.  **Unsupervised Clustering (General Test):** A pipeline to group disks by morphology and detect potential planet candidates without labeled data, using a VAE's latent space for structure and its reconstruction residual for anomaly detection.
2.  **Ring-Aware Reconstruction (Image Task):** A specialized "DiskVAE" architecture designed to reconstruct disk images with high fidelity, preserving faint rings and planet signatures that standard autoencoders often blur out.

---

## Task Name
**Evaluation Tests for ML4Sci GSoC 2026**

### Task
I have developed two comprehensive notebooks solving the **General Test** (unsupervised clustering) and the **Image Task** (high-fidelity reconstruction). Over the past period, I built custom VAE architectures from scratch, designed domain-specific loss functions, and implemented interactive visualization tools to analyze the results.

---

## Solution Notebooks

### 1. General Test: Unsupervised Clustering
*   **Notebook:** [general_test/General_Test.ipynb](general_test/General_Test.ipynb)
*   **Goal:** Cluster diverse disk images and detect anomalies (planets).
*   **Approach:** 
    *   Trained a VAE to reconstruct *blurred* inputs, forcing the latent space to capture global morphology (rings, gaps, inclination) rather than fine details.
    *   **Planet Detection:** Calculated residuals (`Input - Reconstruction`) to isolate compact sources. Used Laplacian of Gaussian (LoG) filtering to detect planets as anomalies.
    *   **Clustering:** Hierarchical approach—first branching by detected planet count, then sub-clustering by morphological similarity using Gaussian Mixture Models (GMM) on the latent vectors.

### 2. Image Task: Ring-Aware Reconstruction
*   **Notebook:** [image_task/Image_Test.ipynb](image_task/Image_Test.ipynb)
*   **Goal:** Reconstruct disk images while preserving sharp rings and faint planet signatures.
*   **Approach:** 
    *   **DiskVAE Architecture:** Custom U-Net-like VAE with **Radial Conditioning** (injecting distance maps to inform the model of disk geometry) and **Bottleneck Self-Attention** for global coherence.
    *   **Physics-Informed Losses:** Replaced standard MSE with a weighted objective:
        *   **Attention Maps:** Pre-computed "clean" (structure) and "pointness" (planet) maps weight the loss, forcing the model to focus on rings and planets.
        *   **Azimuthal Smoothness & Radial Profile:** Enforce physical priors of disk symmetry.
    *   **Joint Training:** A planet detection head is trained simultaneously with the reconstruction task.

---

## Run Instructions
Both notebooks are self-contained. You can run them locally or in Google Colab (with updated paths).

1.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy matplotlib scipy scikit-learn astropy tqdm ipywidgets pytorch_msssim
    ```
2.  **Dataset:** Ensure `continuum_data_subset/` is present in the root directory.
3.  **Execution:** Open the respective notebook and run all cells. Pre-trained weights (`task2.pth` for the Image Task) will be loaded automatically if present; otherwise, set `TRAIN_FROM_SCRATCH = True` to retrain.

---

## Gallery

### Pipeline Overviews
| General Test Pipeline | Image Task Architecture |
| :---: | :---: |
| ![General Test Pipeline](general_test/general_test_pipeline_v2%20(1).svg) | ![Image Task Architecture](image_task/diskvae_full_architecture3.svg) |

### Results (Image Task)
*High-fidelity reconstruction preserving the faint planet candidate on the right.*
![Image Task Results](image_task/resultplots.png)

### Clustering (General Test)
*Disks grouped by morphological similarity (ring count, gap width, size).*
![Clustering Results](general_test/clustermap.png)

---

## Approaching the Task

I followed a systematic research-based approach for both tasks:

1.  **Domain Understanding:** I studied the properties of ALMA continuum images—specifically how dynamic range (flux) and geometric priors (radial symmetry) could be leveraged.
2.  **Preprocessing:**
    *   **Arcsinh Stretch:** Essential for compressing the high dynamic range of astronomical data to make faint rings visible to the model.
    *   **Radial Subtraction:** For the General Test, I subtracted the median radial profile to isolate planets from the background disk structure.
3.  **Architecture Design:**
    *   For the **Image Task**, I realized standard CNNs struggle with the "thin ring" structure. I introduced **Radial Conditioning**—explicitly feeding polar coordinate info—which significantly improved ring sharpness.
4.  **Loss Engineering:**
    *   MSE alone produced blurry results. I implemented **SSIM** (structural similarity) and **Gradient Loss** to preserve edges.
    *   To catch planets, I used **Attention-Weighted Loss**, giving 5000x weight to high-frequency "pointness" regions.
5.  **Evaluation:**
    *   Beyond MSE, I used **MS-SSIM** (Multi-Scale SSIM) as a primary metric for perceptual quality.
    *   Interactive widgets were built to manually inspect hundreds of test results rapidly.

---

## Evaluation Metrics

### Image Task Results
The DiskVAE model achieves state-of-the-art reconstruction fidelity on the hold-out set:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **MSE** | `2.86e-5` | Extremely low pixel-wise error |
| **MS-SSIM** | `0.9852` | High structural preservation |
| **SSIM** | `0.9856` | Excellent local contrast retention |

### General Test Results
*   **Cluster Purity:** Successfully separates "clean" disks from "structured/multi-ring" disks.
*   **Anomaly Detection:** The residual-based method effectively highlights point sources enabling automatic planet candidate flagging.

---

## Future Endeavours

1.  **Transformers for Global Context:**
    Replacing the convolutional bottleneck with a Vision Transformer (ViT) block could better capture long-range dependencies, such as spiral arms that span the entire disk, which CNNs sometimes fragment.
    
2.  **Self-Supervised Contrastive Learning:**
    Instead of pure reconstruction, using SimCLR or MoCo could learn more robust representations for clustering, making the embedding space invariant to rotation and noise without needing explicit augmentations.

3.  **3D Radiative Transfer Integration:**
    Future models could incorporate a physical decoder (like a differentiable RADMC-3D) to regress physical parameters (dust mass, grain size) directly from the image, rather than just pixel intensities.
