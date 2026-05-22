# ECG Arrhythmia Classification with Deep Learning

This repository contains a deep learning project for ECG classification. The work is organized in two parts.

1. **MIT-BIH baseline**: a first heartbeat-level 1D-CNN baseline using raw MIT-BIH ECG records.
2. **PTB-XL multivariate extension**: a modular PyTorch pipeline for multi-label ECG diagnostic classification using 12-lead ECG signals and clinical metadata.

The final PTB-XL model uses a shared-loss fusion architecture:

```text
12-lead ECG signal  -> 1D-CNN -> ECG feature vector
Clinical metadata   -> MLP    -> metadata feature vector
ECG vector + metadata vector -> final classifier -> diagnostic labels
```

The PTB-XL task is treated as a **multi-label classification problem** with five diagnostic superclasses:

```text
NORM, MI, STTC, CD, HYP
```

Therefore, the model uses `BCEWithLogitsLoss` rather than `CrossEntropyLoss`.

---

## Repository structure

```text
Deep-Learning/
├── main.py                         # Main training/evaluation entry point
├── soft_voting.py                  # Prediction-level fusion benchmark
├── dataset/                        # PTB-XL Dataset and DataLoader logic
├── preprocessing/                  # Signal, metadata and label preprocessing
├── net/                            # CNN, MLP and fusion architectures
├── utility/                        # Metrics, plots and checkpoints
├── notebooks/
│   └── 01_mitbih_baseline.ipynb    # Initial MIT-BIH baseline notebook
├── exports/
│   └── 01_mitbih_baseline.html     # HTML export of the MIT-BIH baseline
├── runs/                           # Final experiment outputs
├── report_overleaf/                # LaTeX report project
├── data/                           # Local dataset folder, not tracked
├── cache/                          # Tensor cache folder, not tracked
├── requirements.txt
└── README.md
```

---

## Dataset

### MIT-BIH

The MIT-BIH baseline notebook loads the MIT-BIH Arrhythmia Database through `wfdb`, extracts heartbeat windows around annotations, and trains a 1D-CNN classifier.

### PTB-XL

The PTB-XL dataset is **not included** in this repository because of its size. The raw dataset and the tensor cache must stay local and are ignored by Git.

Expected local structure:

```text
data/
└── ptb-xl/
    └── 1.0.3/
        ├── ptbxl_database.csv
        ├── scp_statements.csv
        ├── records100/
        └── records500/
```

The code uses the 100 Hz records by default. Each ECG input has shape:

```text
(batch_size, 12, 1000)
```

where the 12 channels correspond to the 12 ECG leads.

The metadata branch uses the following clean clinical variables:

```text
age, sex, height, weight
```

Other columns in `ptbxl_database.csv` were not used directly because many of them are identifiers, file paths, reports, fold information or diagnostic code fields that could introduce leakage.

### Downloading PTB-XL

The official source is PhysioNet. If the connection is acceptable, the dataset can be downloaded with WFDB:

```bash
python -c "import wfdb; wfdb.dl_database('ptb-xl', dl_dir='data/ptb-xl/1.0.3')"
```

However, PhysioNet can be slow depending on the connection. A practical alternative is to download the PTB-XL archive from Kaggle or another mirror containing the same PTB-XL 1.0.3 files. The important point is not the download source, but the final folder structure. After extraction, the folder passed to `--data-dir` must directly contain:

```text
ptbxl_database.csv
scp_statements.csv
records100/
records500/
```

For example, if the extracted dataset is stored in:

```text
C:\Users\<user>\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
```

you can either move its content into `data/ptb-xl/1.0.3/`, or directly pass this folder to the script:

```bash
python -u main.py --data-dir "C:/Users/<user>/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" --model fusion --epochs 2 --batch-size 32 --max-samples 500 --use-cache
```

To verify that the folder is correct, check that `ptbxl_database.csv` is directly inside the selected `--data-dir` folder.

### Tensor cache

The first PTB-XL run with `--use-cache` reads the raw WFDB files and stores preprocessed tensors in `cache/`. This is slow only the first time. The cache is not a trained model; it only contains preprocessed ECG tensors, metadata tensors and labels. New models can still be trained normally from the same cache.

---

## Installation

Install the basic dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

PyTorch installation depends on the machine. For an NVIDIA GPU, install a CUDA-compatible version from the official PyTorch instructions.

Example:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Check CUDA:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Quick debug run

Use a small subset only to check that the code runs:

```bash
python -u main.py --data-dir data/ptb-xl/1.0.3 --model fusion --epochs 2 --batch-size 32 --max-samples 500 --use-cache --debug-batches
```

This is **not** representative of final performance.

---

## Full PTB-XL runs

### Shared-loss fusion model

```bash
python -u main.py --data-dir data/ptb-xl/1.0.3 --model fusion --epochs 20 --batch-size 64 --use-cache --tune-thresholds --use-scheduler --patience 6
```

### ECG-only benchmark

```bash
python -u main.py --data-dir data/ptb-xl/1.0.3 --model ecg_only --epochs 10 --batch-size 64 --use-cache --tune-thresholds
```

### Metadata-only benchmark

```bash
python -u main.py --data-dir data/ptb-xl/1.0.3 --model metadata_only --epochs 10 --batch-size 64 --use-cache --tune-thresholds
```

### Soft-voting benchmark

After running ECG-only and metadata-only:

```bash
python -u soft_voting.py --ecg-run runs/20260521_160125_ecg_only --metadata-run runs/20260521_154953_metadata_only --tune-alpha --tune-thresholds
```

---

## Final PTB-XL results

All final runs used the official PTB-XL split:

```text
Train:      17418 ECGs
Validation:  2183 ECGs
Test:        2198 ECGs
```

| Model | Macro-F1 | Micro-F1 | Exact match accuracy |
|---|---:|---:|---:|
| Metadata-only MLP | 0.4645 | 0.4909 | 0.2234 |
| ECG-only 1D-CNN | 0.7091 | 0.7497 | 0.5528 |
| Soft voting | 0.7084 | 0.7491 | 0.5523 |
| Shared-loss fusion | **0.7108** | **0.7536** | **0.5660** |

The shared-loss fusion model achieved the best overall performance, although the improvement over ECG-only is modest. This suggests that most of the discriminative information comes from the ECG signal, while metadata provides a small complementary clinical context.

The soft-voting benchmark selected an optimal validation alpha of `0.9`, meaning that the best prediction-level ensemble relied mostly on the ECG-only model and only weakly on metadata-only predictions. This supports the choice of feature-level fusion as the final architecture.

---

## Experiment folders

```text
runs/20260521_154229_fusion
    Final V4 shared-loss fusion run.

runs/20260521_160125_ecg_only
    Final V4 ECG-only benchmark.

runs/20260521_154953_metadata_only
    Metadata-only MLP benchmark.

runs/20260521_160422_soft_voting
    Soft-voting benchmark.
```

Each run contains:

```text
parameters/     # arguments used for the run
scores/         # JSON metrics and saved predictions
outputs/        # training curves and precision-recall curves
```

---

## Notes on exact match accuracy

PTB-XL is multi-label. Exact match accuracy is strict: a prediction is counted as correct only if all five labels are predicted correctly at the same time. Macro-F1, micro-F1 and precision-recall curves are therefore more informative for this task.

---

## Report

The LaTeX report is available in:

```text
report_overleaf/
```

It can be imported directly into Overleaf.
