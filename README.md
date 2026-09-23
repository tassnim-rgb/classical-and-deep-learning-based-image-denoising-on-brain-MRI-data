# Classical and Deep Learning Based Image Denoising on Brain MRI Data

A comparative study of one classical and two deep learning denoising methods, applied to brain MRI scans corrupted with Gaussian, speckle, and salt and pepper noise.

The project runs the full pipeline in a single Jupyter notebook: preprocessing, synthetic noise injection, training two deep models from scratch (DnCNN and SwinIR), denoising with all three methods, and a quantitative comparison using PSNR, SSIM, and MSE. The headline result is honest and perhaps surprising: on this benchmark the classical algorithm, BM3D, outperforms the trained deep models on most noise conditions.

> All quantitative numbers reported here are reproduced from the notebook's saved outputs (the summary table printed in the evaluation section).

## Overview

This work was developed as part of a university course on digital image analysis and processing (National Higher School of Mathematics). It answers a practical question: for denoising brain MRI images, do classical algorithms still hold their own against modern deep learning models?

The notebook is fully executed end to end: it downloads setup data, loads the dataset, injects noise, trains DnCNN and SwinIR, runs BM3D, and evaluates every combination of method and noise type. The notebook was executed on a Kaggle GPU, with memory-saving measures (mixed precision, gradient accumulation) required to fit two models in VRAM.

## Problem

Magnetic resonance images are corrupted by noise during acquisition, which can obscure fine anatomical structures important for diagnosis. Denoising is therefore a standard preprocessing step. Three families of methods exist: classical filtering, convolutional neural networks, and transformer-based models. This project compares one representative of each family on a small, realistic medical imaging dataset, across three common noise models.

## Approach

The pipeline, as implemented in the notebook:

```
Brain MRI scans (253 JPG images, Kaggle dataset)
                    |
                    v
    Preprocessing: grayscale, resize to 128x128, float64 in [0, 1]
                    |
        +-----------+-----------+
        |                       |
        v                       v
  Clean images          Noise injection
                         (Gaussian | Speckle | Salt and Pepper)
        |                       |
        +-----------+-----------+-----------+
                    |                       |
                    v                       v
              BM3D (classical)      DnCNN / SwinIR (trained)
                    |                       |
                    +-----------+-----------+
                                |
                                v
                  Evaluation: PSNR, SSIM, MSE
                  (mean over the 53-image test set)
```

Step by step:

1. **Preprocessing.** Each image is loaded with PIL, converted to grayscale, resized to 128 x 128, and normalized to float64 values in [0, 1].
2. **Noise injection.** Synthetic noise is added per the three noise models described below, with fixed noise levels.
3. **Denoising.** The same noisy images are denoised by BM3D (zero-shot, CPU), DnCNN, and SwinIR (both trained from scratch on Gaussian noise).
4. **Evaluation.** Mean PSNR, SSIM, and MSE are computed over the held-out evaluation set for every (noise type, method) pair, and summarized in tables and charts.

## Dataset

The notebook uses the **Brain MRI Images for Brain Tumor Detection** dataset published on Kaggle by Navoneel Chakrabarty.

- 253 brain MRI scans in JPEG format: 155 tumor images (yes class) and 98 non-tumor images (no class), organized in class folders.
- All images are resized to 128 x 128 grayscale for this project.
- The first 200 images are used to train the neural networks and the remaining 53 form the evaluation set (as described in the notebook).
- The dataset itself is not included in this repository. The notebook reads from the Kaggle-mounted path `DATASET_ROOT = '/kaggle/input/datasets/navoneel/brain-mri-images-for-brain-tumor-detection'`; running it locally requires downloading the dataset and updating this path.

## Noise Models

The notebook defines three standard synthetic noise models. All operate on clean float64 arrays in [0, 1] and clip outputs back to that range.

- **Gaussian noise.** Additive white noise drawn from N(0, sigma^2) with sigma = 0.1. This is the most common assumption in denoising and approximates Rician MRI noise at high signal-to-noise ratio.
- **Speckle noise.** Multiplicative noise of the form `noisy = image + image * N(0, sigma^2)` with sigma = 0.1, typical of coherent imaging systems.
- **Salt and pepper noise.** A random 5% of pixels are replaced by extreme values (half set to 0, half to 1), modeling impulse corruption.

## Methods

- **BM3D (Dabov et al., 2007):** Block-Matching and 3D Filtering, the classical baseline. It groups similar patches across the image and filters them jointly in a 3D transform domain in a two-step pipeline (basic estimate followed by Wiener filtering). It is zero-shot (no training, no parameters) and runs on CPU.
- **DnCNN (Zhang et al., 2017):** A 17-layer residual convolutional network (64 channels) that learns to predict the noise residual rather than the clean image directly; the predicted noise is subtracted from the input. It is built as a plain `nn.Sequential` module and trained for 30 epochs with Adam (learning rate 1e-3), MSE loss, and cosine annealing to 1e-5. Approximately 556k parameters.
- **SwinIR (Liang et al., 2021):** A transformer-based image restoration network using shifted-window self-attention, implemented procedurally in the notebook (window partition, shifted windows, relative position bias) and assembled with `OrderedDict`. It is trained for 30 epochs with AdamW (learning rate 5e-4, weight decay 1e-4), automatic mixed precision, and gradient accumulation (batch size 4, 8 accumulation steps for an effective batch of 32). Approximately 49k parameters.

Both deep models are trained exclusively on Gaussian noise (sigma = 0.1) and evaluated on all three noise types, so the speckle and salt and pepper evaluations probe cross-noise generalization.

## Evaluation Metrics

All metrics are computed with scikit-image against the clean reference image.

- **PSNR (Peak Signal-to-Noise Ratio):** the ratio between the maximum possible signal power and the mean squared error, expressed in decibels. Higher is better.
- **SSIM (Structural Similarity Index):** a value in [0, 1] comparing luminance, contrast, and structure between two images; higher values indicate higher perceived similarity.
- **MSE (Mean Squared Error):** the average squared pixel difference between the denoised and clean images; lower is better.

## Results

Mean metrics over the 53-image evaluation set, exactly as printed by the notebook's summary table.

| Noise Type | Method | PSNR (dB) | SSIM | MSE |
|------------|--------|-----------|------|---------|
| Gaussian   | BM3D   | **28.56** | 0.7286 | 0.001427 |
| Gaussian   | DnCNN  | 26.22 | 0.6103 | 0.002432 |
| Gaussian   | SwinIR | 27.10 | **0.7418** | 0.002008 |
| Speckle    | BM3D   | **31.75** | **0.9091** | **0.000698** |
| Speckle    | DnCNN  | 29.30 | 0.7442 | 0.001224 |
| Speckle    | SwinIR | 29.66 | 0.7983 | 0.001196 |
| Salt and Pepper | BM3D | 20.86 | 0.5683 | 0.008376 |
| Salt and Pepper | DnCNN | 21.17 | 0.4879 | 0.007681 |
| Salt and Pepper | SwinIR | **23.93** | **0.5814** | **0.004081** |

(Bold marks the best value per metric within each noise type.)

Key takeaways:

- **Classical BM3D outperforms both deep models on Gaussian and speckle noise**, taking the best PSNR in both cases and the best SSIM/MSE on speckle.
- **SwinIR leads on salt and pepper noise** (best PSNR, SSIM, and MSE), and also has the best SSIM on Gaussian noise.
- **DnCNN is the weakest of the three** in this setup, which is consistent with its smaller effective receptive field and the modest training budget.

### Why BM3D wins here

This result is not evidence that classical methods are generally superior. It reflects the specifics of this benchmark:

- **BM3D is a specialized, zero-shot denoiser.** Its non-local block matching and Wiener filtering are near-optimal for additive Gaussian noise with known sigma, and with zero parameters it cannot overfit.
- **The dataset is small.** The deep models train on only 200 low-resolution (128 x 128) images, while DnCNN and SwinIR are data-hungry by design.
- **The training budget is modest.** Both networks run for 30 epochs only, with no hyperparameter search, and are trained on Gaussian noise alone.
- **BM3D is given its assumed noise level** (sigma = 0.1) while the deep models must infer it.

In short, on this small benchmark the gap is less about deep learning being inadequate and more about the classical algorithm being perfectly matched to the task.

## Technologies

- Python
- PyTorch and TorchVision (deep model definition and training)
- scikit-image (noise generation, metrics, BM3D via the `bm3d` package)
- NumPy, pandas (array processing and result aggregation)
- Pillow (image I/O)
- Matplotlib (visualization)

## Installation

```bash
pip install -r requirements.txt
```

PyTorch and TorchVision wheels should be selected to match your CUDA version (see https://pytorch.org). The notebook was executed on a Kaggle GPU, but the code falls back to CPU when CUDA is unavailable; training is slow on CPU.

## Usage

1. Obtain the dataset on Kaggle by adding the "Brain MRI Images for Brain Tumor Detection" dataset to a notebook, or download it and update `DATASET_ROOT` in the notebook.
2. Open `Main.ipynb` in Jupyter, Kaggle, or VS Code.
3. Run all cells top to bottom. The notebook performs noise injection, trains DnCNN and SwinIR, runs BM3D, and prints the comparison tables and charts.
4. Training checkpoints are saved as `dncnn_best.pth` and `swinir_best.pth`; figures are saved as PNG files in the working directory. These artifacts are git-ignored.

## Project Structure

```
.
├── Main.ipynb        # Full pipeline: preprocessing, noise, training, evaluation
├── requirements.txt  # Python dependencies
├── README.md         # This document
├── LICENSE           # MIT license
└── .gitignore
```

## Limitations

- **Single dataset.** All results come from one small dataset (253 brain MRI scans at 128 x 128 resolution); conclusions may not transfer to other image domains or resolutions.
- **Gaussian-only training.** The deep models are trained on Gaussian noise but evaluated on all three noise types, so their speckle and salt and pepper scores reflect limited cross-noise generalization.
- **Modest training budget.** Both networks run for 30 epochs with fixed hyperparameters and no search; the notebook's own discussion notes that more data and epochs are expected to narrow the gap to BM3D.
- **BM3D noise assumption.** BM3D is always called with the Gaussian sigma, even for speckle and salt and pepper inputs, which penalizes it on those two noise types.
- **Aggregate metrics only.** Reported numbers are means over the evaluation set; no standard deviations or per-image distributions are reported.

## Future Work

- Train on more data and higher resolution (for example, 256 x 256 or 512 x 512 scans) with a longer schedule and warm-up.
- Use blind-denoising variants, such as DnCNN-B or noise-level-conditioned SwinIR, and train on multiple noise types.
- Evaluate on realistic Rician MRI noise in addition to synthetic models.
- Complement MSE with a perceptual loss such as LPIPS.

## References

- K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian. "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering." IEEE Transactions on Image Processing, vol. 16, no. 8, 2007. (BM3D)
- K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising." IEEE Transactions on Image Processing, vol. 26, no. 7, 2017. (DnCNN)
- J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte. "SwinIR: Image Restoration Using Swin Transformer." IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021. (SwinIR)

## Acknowledgments

Dataset: Navoneel Chakrabarty, "Brain MRI Images for Brain Tumor Detection," Kaggle. This work was supervised by Dr. Amel BOUCHEMHA at the National Higher School of Mathematics.