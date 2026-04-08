# Pneumonia Detection from Chest X-ray

Deep learning project for binary chest X-ray classification:

- Class 0: Normal
- Class 1: Pneumonia

The repository includes:

- Training pipeline for multiple CNN backbones (ResNet18, EfficientNet-B0, DenseNet121)
- Evaluation with standard classification metrics
- Grad-CAM visual explanation utilities
- Streamlit web app for sample browsing and image prediction
- EDA scripts for dataset checks and exploratory analysis

## Project Structure

```text
.
├── app/
│   ├── main.py
│   └── pages/
│       ├── 1_Samples.py
│       └── 2_Predict.py
├── data/
│   ├── dataset.py
│   └── transforms.py
├── eda/
│   ├── brightness_analysis.py
│   ├── data_distribution.py
│   ├── image_samples.py
│   ├── image_stats.py
│   └── leakage_check.py
├── images/
│   ├── normal/
│   └── pneumonia/
├── models/
│   ├── densenet.py
│   ├── efficientnet.py
│   └── resnet.py
├── outputs/
│   └── checkpoints/
├── training/
│   ├── engine.py
│   └── train.py
├── utils/
│   ├── gradcam.py
│   └── metrics.py
├── requirements.txt
└── README.md
```

## Dataset Layout

Training code expects this exact folder layout under `dataset/`:

```text
dataset/
├── train/
│   ├── normal/
│   └── pneumonia/
├── val/
│   ├── normal/
│   └── pneumonia/
└── test/
    ├── normal/
    └── pneumonia/
```

Image files can be `.png`, `.jpg`, or `.jpeg`.

## Environment Setup

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For full training metrics and EDA scripts, also install:

```bash
pip install scikit-learn tqdm pandas matplotlib
```

## Train Models

Run training from project root:

```bash
python -m training.train
```

What happens:

- Loads grayscale chest X-rays via custom dataset class
- Applies augmentation and normalization transforms
- Trains each backbone for 5 epochs
- Computes metrics (accuracy, precision, recall, F1)
- Saves checkpoints to `outputs/checkpoints/`

Expected checkpoints:

- `outputs/checkpoints/resnet18.pt`
- `outputs/checkpoints/efficientnet_b0.pt`
- `outputs/checkpoints/densenet121.pt`

## Run Streamlit App

Start the app:

```bash
streamlit run app/main.py
```

The app includes two pages:

- Samples: preview example images from `images/normal` and `images/pneumonia`
- Predict: upload a chest X-ray and get:
  - Predicted class (Normal or Pneumonia)
  - Confidence score
  - Grad-CAM heatmap overlay

Note: prediction page loads `outputs/checkpoints/densenet121.pt`. Make sure this file exists before inference.

## Run EDA Scripts

From project root, execute any script below:

```bash
python -m eda.data_distribution
python -m eda.image_stats
python -m eda.brightness_analysis
python -m eda.image_samples
python -m eda.leakage_check
```

These scripts help inspect:

- Class counts across train/val/test
- Image shape statistics
- Brightness distribution
- Random sample visualization
- Potential train/test filename leakage

## Models

Implemented backbones:

- ResNet18
- EfficientNet-B0
- DenseNet121

All models are adapted for single-channel (grayscale) input and binary output.

## Explainability

`utils/gradcam.py` provides Grad-CAM visualization used by the Streamlit prediction page to highlight attended regions.

## Troubleshooting

- If training fails with missing packages, install optional dependencies listed above.
- If Streamlit prediction page shows model load error, verify checkpoint path and filename.
- If dataset loading fails, verify directory names are exactly `normal` and `pneumonia`.

## Disclaimer

This project is for educational and research purposes only. It is not a certified medical device and must not be used as a standalone clinical decision system.