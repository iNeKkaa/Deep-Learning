# ECG Arrhythmia Classification with Deep Learning

This project focuses on heartbeat classification using the MIT-BIH Arrhythmia Database.

The objective is to build a complete deep learning pipeline starting from raw ECG records. The project includes ECG signal loading, heartbeat extraction, preprocessing, training of a 1D Convolutional Neural Network, and evaluation with standard classification metrics.

## Project overview

The pipeline follows these steps:

1. Load raw ECG records and expert annotations from the MIT-BIH Arrhythmia Database.
2. Extract heartbeat windows around annotated beat locations.
3. Normalize each extracted heartbeat.
4. Map the original MIT-BIH annotation symbols into five heartbeat classes.
5. Train a 1D-CNN model using PyTorch.
6. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrices.

## Project structure

```text
deep-learning-ecg/
│
├── notebooks/
│   └── 01_mitbih_baseline.ipynb
│
├── models/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

## Dataset

The dataset used in this project is the MIT-BIH Arrhythmia Database.

The ECG records are loaded directly with the `wfdb` Python library, so no manual dataset download is required.

Each ECG recording contains expert annotations indicating the position and type of heartbeats. Around each annotated heartbeat, a fixed-size signal window is extracted and used as input for the neural network.

## Classes

The original MIT-BIH heartbeat annotations are grouped into five classes:

| Label | Class |
|---:|---|
| 0 | Normal beat |
| 1 | Supraventricular ectopic beat |
| 2 | Ventricular ectopic beat |
| 3 | Fusion beat |
| 4 | Unknown / paced / unclassified beat |

## Local setup

First, clone or download the project, then move into the project folder.

Install the basic dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

PyTorch is not included in `requirements.txt` because the installation depends on the machine and GPU configuration.

For local GPU training with an NVIDIA GPU, install PyTorch with CUDA:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If this version does not work, use:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

To check if PyTorch detects the GPU, run:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Expected output when CUDA is correctly configured:

```text
True
NVIDIA GeForce RTX 3090
```

## Google Colab

The notebook is also compatible with Google Colab.

Before running the notebook, enable GPU acceleration:

```text
Runtime > Change runtime type > GPU
```

The notebook contains a small setup cell that installs `wfdb` only when it detects a Colab environment.

No local virtual environment is required on Colab.

## Requirements

The basic project dependencies are listed in `requirements.txt`:

```txt
numpy
matplotlib
pandas>=2.2.3
wfdb
tqdm
ipykernel
```

PyTorch must be installed separately depending on the execution environment.

## Notebook

The main notebook is:

```text
notebooks/01_mitbih_baseline.ipynb
```

It contains:

- environment check;
- MIT-BIH record loading;
- heartbeat extraction from raw ECG signals;
- class distribution analysis;
- signal visualization;
- PyTorch dataset and dataloader creation;
- 1D-CNN model definition;
- training loop;
- validation loop;
- final test evaluation;
- confusion matrix and normalized confusion matrix.

## Model

The trained PyTorch model is saved in the `models/` folder.

Example:

```text
models/best_mitbih_cnn1d.pt
```

The model file is ignored by Git by default because it can become large and depends on the training run.

## Evaluation metrics

The model is evaluated with:

- accuracy;
- precision;
- recall;
- F1-score;
- macro-F1 score;
- confusion matrix;
- normalized confusion matrix.

Since the dataset is imbalanced, accuracy alone is not enough. Macro-F1 and per-class recall are especially important, because the model must not only classify normal heartbeats correctly but also detect minority arrhythmia classes.

## Notes

The first baseline uses a 1D-CNN because ECG signals are one-dimensional time series. Convolutional layers are well suited for extracting local temporal patterns such as peaks, waveform shapes, and rhythm variations.

Further improvements can include:

- testing different CNN architectures;
- tuning learning rate, batch size, and number of epochs;
- adding an LSTM or GRU model for comparison;
- using a larger ECG dataset such as PTB-XL;
- working with multi-lead ECG signals instead of a single ECG channel.