# ECG Classification with Deep Learning

This repository contains a deep learning project for ECG classification.

The project is organized in two stages:

1. **MIT-BIH baseline**
   A first heartbeat-level 1D-CNN baseline using raw MIT-BIH ECG records.

2. **PTB-XL 12-channel extension**
   A modular PyTorch pipeline for multi-label ECG diagnostic classification using full 10-second ECG recordings and clinical metadata.

The final PTB-XL model uses a **shared-loss feature-level fusion architecture**:

```text
12-channel ECG signal -> 1D-CNN -> ECG feature vector
Clinical metadata     -> MLP    -> metadata feature vector

ECG features + metadata features
        -> final classifier
        -> 5 diagnostic logits
```

The PTB-XL task is treated as a **multi-label classification problem** over five diagnostic superclasses:

```text
NORM, MI, STTC, CD, HYP
```

Because several diagnostic groups can be present in the same ECG recording, the model uses `BCEWithLogitsLoss` rather than `CrossEntropyLoss`.

---

## Main idea

The main question of the project is:

> Can static clinical metadata improve 12-channel ECG classification, and is feature-level fusion better than prediction-level soft voting?

The final result is nuanced:

* ECG-only classification is already strong.
* Metadata alone is clearly insufficient.
* Shared-loss fusion gives the best final metrics.
* The improvement over ECG-only is real but modest.

---

## Repository structure

```text
Deep-Learning/
├── main.py                         # Main PTB-XL training/evaluation entry point
├── soft_voting.py                  # Prediction-level fusion benchmark
├── dataset/                        # PTB-XL Dataset and DataLoader logic
├── preprocessing/                  # Signal, metadata and label preprocessing
├── net/                            # CNN, MLP and fusion architectures
├── utility/                        # Metrics, plots and checkpoints
├── notebooks/
│   └── 01_mitbih_baseline.ipynb    # MIT-BIH baseline notebook
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

## Datasets

### MIT-BIH baseline

The MIT-BIH baseline is a **heartbeat-level classification task**.

The notebook:

1. loads raw ECG records and expert annotations;
2. extracts fixed-size windows around annotated heartbeats;
3. normalizes each heartbeat window;
4. maps MIT-BIH annotation symbols into five heartbeat classes;
5. trains a 1D-CNN classifier.

Each example is one heartbeat window, and each heartbeat has exactly one label. Therefore, this baseline is a **multi-class single-label** classification task.

The MIT-BIH model uses `CrossEntropyLoss`.

---

### PTB-XL dataset

The main project uses PTB-XL.

The dataset is not included in this repository because of its size. The raw dataset and tensor cache must stay local and are ignored by Git.

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

The code uses the 100 Hz records by default. Each ECG recording lasts 10 seconds, so each channel contains 1000 time samples.

The ECG input shape is:

```text
(batch_size, 12, 1000)
```

where:

* `batch_size` is the number of ECG recordings processed together;
* `12` is the number of ECG channels;
* `1000` is the number of time samples per channel.

Unlike MIT-BIH, PTB-XL is not beat-level classification. The model receives the full 10-second ECG recording and predicts diagnostic labels for the whole recording.

---

## Metadata

The metadata branch uses four static clinical variables:

```text
age, sex, height, weight
```

These variables are patient-level information. They are not temporal ECG channels.

Other columns from `ptbxl_database.csv` are not used directly because many of them are identifiers, file paths, reports, fold information or diagnostic code fields. Diagnostic code fields are excluded to avoid **label leakage**, since they are directly related to the target labels.

---

## Installation

Install the Python dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

PyTorch installation depends on your machine. For an NVIDIA GPU, install a CUDA-compatible PyTorch version from the official PyTorch website.

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

## Downloading PTB-XL

The official source is PhysioNet. If the connection is stable, PTB-XL can be downloaded with WFDB:

```bash
python -c "import wfdb; wfdb.dl_database('ptb-xl', dl_dir='data/ptb-xl/1.0.3')"
```

After downloading or extracting the dataset, make sure the folder passed to `--data-dir` directly contains:

```text
ptbxl_database.csv
scp_statements.csv
records100/
records500/
```

For example:

```bash
python -u main.py --data-dir data/ptb-xl/1.0.3 --model fusion --epochs 2 --batch-size 32 --max-samples 500 --use-cache
```

---

## Tensor cache

The first PTB-XL run with `--use-cache` reads the raw WFDB files and stores preprocessed tensors in `cache/`.

This can be slow the first time, but later runs are much faster.

The cache is not a trained model. It only contains preprocessed ECG tensors, metadata tensors and labels.

---

## Quick debug run

Use a small subset only to verify that the code runs:

```bash
python -u main.py \
  --data-dir data/ptb-xl/1.0.3 \
  --model fusion \
  --epochs 2 \
  --batch-size 32 \
  --max-samples 500 \
  --use-cache \
  --debug-batches
```

This run is only for debugging and is not representative of final performance.

---

## Full PTB-XL experiments

### Shared-loss fusion model

```bash
python -u main.py \
  --data-dir data/ptb-xl/1.0.3 \
  --model fusion \
  --epochs 20 \
  --batch-size 64 \
  --use-cache \
  --tune-thresholds \
  --use-scheduler \
  --patience 6
```

### ECG-only benchmark

```bash
python -u main.py \
  --data-dir data/ptb-xl/1.0.3 \
  --model ecg_only \
  --epochs 10 \
  --batch-size 64 \
  --use-cache \
  --tune-thresholds
```

### Metadata-only benchmark

```bash
python -u main.py \
  --data-dir data/ptb-xl/1.0.3 \
  --model metadata_only \
  --epochs 10 \
  --batch-size 64 \
  --use-cache \
  --tune-thresholds
```

### Soft-voting benchmark

After running the ECG-only and metadata-only models:

```bash
python -u soft_voting.py \
  --ecg-run runs/20260521_160125_ecg_only \
  --metadata-run runs/20260521_154953_metadata_only \
  --tune-alpha \
  --tune-thresholds
```

Soft voting combines probabilities, not logits:

```text
p_soft = alpha * p_ecg + (1 - alpha) * p_metadata
```

No additional MLP is trained after the weighted average.

---

## Training and evaluation logic

The official PTB-XL split is used:

```text
Folds 1-8 -> training
Fold 9    -> validation
Fold 10   -> test
```

The roles of the splits are:

* **training set**: used to update the model weights;
* **validation set**: used for early stopping, threshold tuning and alpha selection;
* **test set**: used only for final evaluation.

For PTB-XL, the model outputs five logits. During training, these logits are passed directly to `BCEWithLogitsLoss`.

During evaluation:

```text
logits -> sigmoid -> probabilities -> class-specific thresholds -> labels
```

The final fusion thresholds were:

```text
NORM: 0.65
MI:   0.70
STTC: 0.50
CD:   0.65
HYP:  0.60
```

---

## Final PTB-XL results

All final results were obtained using the official PTB-XL split:

```text
Training:   17,418 ECGs
Validation:  2,183 ECGs
Test:        2,198 ECGs
```

| Model              |   Macro-F1 |   Micro-F1 | Exact match accuracy |
| ------------------ | ---------: | ---------: | -------------------: |
| Metadata-only MLP  |     0.4645 |     0.4909 |               0.2234 |
| ECG-only 1D-CNN    |     0.7091 |     0.7497 |               0.5528 |
| Soft voting        |     0.7084 |     0.7491 |               0.5523 |
| Shared-loss fusion | **0.7108** | **0.7536** |           **0.5660** |

The shared-loss fusion model obtained the best final performance, but the improvement over ECG-only is modest. This suggests that the ECG signal remains the dominant source of information, while metadata provides a small complementary clinical contribution.

The soft-voting benchmark selected:

```text
alpha = 0.9
```

This means that the best prediction-level ensemble relied mostly on ECG-only predictions and only weakly on metadata-only predictions.

---

## Experiment folders

```text
runs/20260521_154229_fusion
    Final shared-loss fusion run.

runs/20260521_160125_ecg_only
    Final ECG-only benchmark.

runs/20260521_154953_metadata_only
    Metadata-only MLP benchmark.

runs/20260521_160422_soft_voting
    Soft-voting benchmark.
```

Each run contains:

```text
parameters/     # arguments used for the run
scores/         # JSON metrics and saved predictions
outputs/        # training curves, PR curves and plots
```

---

## Metrics

PTB-XL is multi-label and imbalanced. Therefore, exact match accuracy alone is not enough.

The main reported metrics are:

* **macro-F1**: computes F1 per class and averages the five scores equally;
* **micro-F1**: pools all true positives, false positives and false negatives before computing F1;
* **per-class F1**: shows which diagnostic groups are easier or harder;
* **precision-recall curves**: show the threshold-dependent precision/recall trade-off.

Exact match accuracy is strict: a prediction is counted as correct only if all five labels are predicted correctly at the same time.

---

## Report

The LaTeX report is available in:

```text
report_overleaf/
```

It follows the required structure:

```text
Introduction
State of the Art
Methodology
Experiments
Results
Discussion
Conclusions
References
```

---

## Main conclusion

The project shows that adding metadata to ECG classification can help, but only modestly.

The ECG signal remains the main predictive source. Metadata alone is weak, but feature-level fusion allows the model to use metadata as a small complementary clinical context.

Future work should focus on:

* better imbalance handling, especially for HYP;
* stronger ECG architectures;
* probability calibration and a deeper analysis of metadata contribution.
