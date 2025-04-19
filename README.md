# ESC_classifier

**Interpretable Audio Classification Using CNNs and Non-negative Matrix Factorization (NMF)**

This project explores the interpretability of convolutional neural networks (CNNs) in the context of environmental sound classification. By combining CNN architectures with Non-negative Matrix Factorization (NMF), we analyze internal network activations to understand how the model learns and distinguishes between various audio classes.

---

## 🔍 Key Features

- **Environmental Sound Classification**: Trained on the [ESC-50 dataset](https://github.com/karoldvl/ESC-50), a benchmark dataset consisting of 50 categories of everyday environmental sounds.
- **CNN-Based Architecture**: Custom convolutional model trained on mel-spectrogram inputs.
- **Interpretability via NMF**: Feature maps from intermediate layers are decomposed using NMF to extract interpretable components.
- **Visual Analysis**: Tools to visualize NMF bases, activations, and their correspondence with spectrogram inputs.

---

## 📁 Repository Contents

- `classifier_nn.ipynb` – Defines and trains the CNN model; includes activation extraction.
- `preprocessing.ipynb` – Handles audio loading, spectrogram generation, and dataset preparation.
- `graphics.py` – Utility functions for visualizing activations and NMF results.
- `main.py` – Main script for model training and NMF-based interpretability.
- `activation_analysis_results/` – Saved visualizations and analysis artifacts.
- `AllModelResults.zip` – Archived model outputs and performance results.
- `writting.pdf` – Report describing methodology, experiments, and results.

---

## 🚀 Getting Started

### 1. Feel free to play around :D
```bash
git clone https://github.com/roshdwivedi/ESC_classifier.git
cd ESC_classifier
