
# A Multi-Stream Spectral Transformer for Universal Synthetic Image Detection

A research project focused on detecting **AI-generated synthetic images** using deep learning and explainable AI techniques. This project implements a baseline detector using **VGG16**, evaluates it on the **CiFake dataset**, and explores **artifact explainability using Grad-CAM and LLM-based interpretation**.

---

# Project Overview

Recent advances in **Generative AI** such as GANs and diffusion models allow the creation of highly realistic synthetic images. While these technologies have many positive applications, they also introduce risks including:

- misinformation
- identity manipulation
- deepfake media
- digital fraud

Traditional fake image detectors often fail to **generalize across different generative models** and provide **little interpretability**.

This project aims to develop a **universal synthetic image detection framework**. We made the following contributions:

- A rigorous mathematical taxonomy of generator-specific artifact signatures spanning spatial, frequency (FFT/DCT/DWT), statistical, and sensor-physics domains.
- Developed a novel transformer architechture called Multi-Stream Spectral Transformer (MSST), the first architecture to unify all five signal domains under a single gated-attention backbone with swappable task heads.
- We introduce Masked Spectral Modeling (MSM) as a self-supervised pre-training objective that teaches the backbone the physics of digital image frequencies without task-specific bias.
- We derive formal proofs for the principal artifact signatures exploited by each detection stream and relate them to their generative-process causes.
- We present comparative analysis demonstrating the superiority of multi-domain approaches over single-domain baselines and document robustness under six real-world image degradation conditions.
- Finally, we integrate interpretable explanations by artifact localization.

# Project Structure
deepfake-detection/
├── src/
│   ├── models/
│   │   ├── msst.py          # Full MSST model
│   │   ├── components.py    # DWT, gating, stream modules
│   │   └── heads.py         # Swappable task heads
│   ├── data/
│   │   ├── dataset.py       # DeepfakeDataset
│   │   └── extractor.py     # ExpertPhysicsExtractor (52 features)
│   └── utils/
│       └── scaler.py        # StandardScaler fitting
├── configs/
│   └── default.yaml         # Training hyperparameters
├── scripts/
│   └── train.py             # Training entry point
├── notebooks/               # Jupyter notebooks
├── checkpoints/             # Saved models (git-ignored)
├── results/                 # Plots and logs (git-ignored)
├── requirements.txt
└── README.md

---
## Citation

```bibtex
@article{msst2025,
  title   = {Multi-Stream Spectral Transformer for
             Robust AI-Generated Image Detection},
  author  = {Md Tanvir Mahmud Prince},
  year    = {2026},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
Finally, copy your existing trained files into the structure:
python# ============================================================
# Copy existing trained artifacts into the new structure
# ============================================================

---

### Training Configuration

| Parameter | Value |
|----------|------|
| Model | VGG16 (Pretrained ImageNet) |
| Image Size | 224 × 224 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |
| Task | Real vs Fake Classification |

---

# Model Evaluation

Model performance is evaluated using:

- Classification accuracy
- Confusion matrix
- Prediction confidence

The confusion matrix helps identify:

- **True Positives** – correctly detected fake images
- **True Negatives** – correctly detected real images
- **False Positives** – real images incorrectly labeled as fake
- **False Negatives** – synthetic images classified as real

These results provide insight into **model reliability and failure cases**.

---

# Explainability with Grad-CAM

Deep neural networks often behave as **black-box models**. To interpret predictions, the project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM highlights the **image regions most responsible for a prediction**.

Example insights:

- irregular textures
- unnatural edges
- high-frequency artifact regions
- inconsistent patterns

These visual explanations help identify **synthetic artifacts introduced by generative models**.

---

# Artifact Interpretation Using LLM

To further improve interpretability, the project integrates a **Large Language Model (LLM)**.

The LLM analyzes:

- model prediction confidence
- Grad-CAM heatmaps
- artifact regions

and produces **human-readable explanations** describing why the image is likely synthetic.

Example explanation:

> “The classifier detected irregular high-frequency textures and unnatural structural patterns consistent with artifacts produced by generative models.”

---

# How to Run the Project

### 1. Install Dependencies

```
pip install tensorflow
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install opencv-python
pip install split-folders
```

---

### 2. Clone Repository

```
git clone https://github.com/yourusername/AI-Synthetic-Image-Detection
cd AI-Synthetic-Image-Detection
```

---

### 3. Run Notebook

```
jupyter notebook
```

Open:

```
Capstone_project_milstone.ipynb
```

and run the cells sequentially.

---

# Future Work

The final phase of this project will implement a **Hybrid Multi-Domain Detection Architecture**.

Planned improvements:

### Frequency Domain Artifact Detection

Using:

- Fast Fourier Transform (FFT)
- Discrete Cosine Transform (DCT)

to detect spectral inconsistencies.

### Hybrid Spatial–Frequency Model

```
Spatial CNN Branch
        +
Frequency CNN Branch
        ↓
Feature Fusion
        ↓
Universal Fake Image Detector
```

### Cross-Generator Generalization

Testing robustness across:

- GAN models
- diffusion models
- autoregressive generators

### Robustness Testing

Evaluate performance under:

- JPEG compression
- Gaussian noise
- image resizing
- adversarial perturbations

---

# Author

**Tanvir Mahmud Prince**

MSc Electrical Engineering

Research interests:

- Artificial Intelligence
- Computer Vision
- Synthetic Media Detection
- AI for Digital Forensics

---

# References

1. Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023.
2. Zhang et al., *Perceptual Artifacts Localization for Image Synthesis Tasks*, ICCV 2023.
3. Jia et al., *Can ChatGPT Detect Deepfakes?*, CVPR 2024.
