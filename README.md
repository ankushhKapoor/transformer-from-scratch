# Transformer From Scratch (PyTorch)

## Overview

This repository contains a **from-scratch implementation of the Transformer architecture** for neural machine translation using **PyTorch**.

The project focuses on:

* Understanding the Transformer architecture at a low level
* Implementing core components manually (attention, masking, decoding, training loop)
* Training and evaluating a bilingual translation model using **OPUS Books**
* Proper evaluation using **SacreBLEU**, **WER**, and **CER**
* Experiment tracking using **TensorBoard**

The implementation is intentionally explicit and educational, avoiding high-level abstractions where possible.

---

## Goals of This Project

* Deep understanding of Transformer internals
* Hands-on experience with sequence-to-sequence models
* Correct evaluation practices in machine translation
* Clean, inspectable, and extensible codebase

---

## Paper Reference

This project is based on the paper **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**

Key ideas implemented from the paper:

* Scaled dot-product attention
* Multi-head attention
* Positional encoding
* Encoder–decoder Transformer architecture
* Label smoothing
* Learning rate warmup and inverse square root decay
* Beam search decoding with length penalty

---

## Installation

Using `uv` (recommended):

```bash
uv sync
```

For GPU training, install a CUDA-compatible PyTorch build.

---

## Project Structure

```
.
├── config.py               # Central configuration file defining model hyperparameters, training settings, and file paths.
├── dataset.py              # Implements the BilingualDataset, padding, token handling, and causal/self-attention masks.
├── model.py                # Full Transformer implementation (encoder, decoder, attention, feed-forward, projection layers).
├── train.py                # End-to-end training pipeline including validation, SacreBLEU evaluation, checkpointing, and LR scheduling.
├── inference.ipynb         # Runs inference on trained checkpoints using greedy and beam search decoding.
├── attention_visual.ipynb  # Visualizes encoder, decoder, and cross-attention weights across layers and heads.
├── weights/                # Saved model checkpoints (per-epoch and best-BLEU models).
├── runs/                   # TensorBoard logs for training loss, BLEU, CER, and WER metrics.
└── README.md               # Project documentation, setup instructions, and usage guide.
```

---

## Dataset

The model is trained on the **OPUS Books** dataset:

* Source: `Helsinki-NLP/opus_books`
* Language pair: English → Italian (configurable)

The dataset provides only a training split, so a **90/10 split** is created manually for training and validation.

### Changing Language Pairs or Dataset

You can change the translation language pair by modifying the following fields in `config.py`:

```python
lang_src = "en"
lang_tgt = "it"
```

These values directly control which language pair is loaded from OPUS.

You may also switch to a **different OPUS dataset** by changing the dataset source inside `train.py`.

### English → Hindi Translation

For **English → Hindi (en–hi)** translation, refer to the **`en-hi` branch** of this repository.

That branch uses:

* Dataset: `opus-100`
* Language pair: `en-hi`

This dataset provides an explicit train/validation split.

Keeping this in a separate branch avoids dataset-specific conditionals in the main training code.

---

## Configuration

All training parameters are defined in **`config.py`**.

Key parameters include:

```python
"batch_size": 8,
"num_epochs": 30,
"lr": 1.0,
"seq_len": 350,
"d_model": 512,
"datasource": 'Helsinki-NLP/opus_books',
"lang_src": "en",
"lang_tgt": "it",
"model_folder": "weights",
"model_basename": "tmodel_",
"preload": None,
"tokenizer_file": "tokenizer_{0}.json",
"experiment_name": "runs/tmodel",
"save_best_only": False,
"save_every": None
```

### Important Note on Batch Size

* The **default batch size in the config is 8**
* During cloud GPU training (L40S), the model was trained using:

```python
batch_size = 80
num_epochs = 40
```

Batch size should be adjusted based on:

* Available GPU memory
* Sequence length
* Model dimension

Smaller batch sizes are recommended for local machines.

---

## Training the Model

To start training:

```bash
uv run train.py
```

During training:

* Tokenizers are built if not already present
* The dataset is split into training and validation
* The Transformer is trained using cross-entropy loss with label smoothing
* Validation is run after each epoch
* BLEU, WER, and CER are logged to TensorBoard
* Checkpoints are saved based on the configuration

___


## Checkpointing and Resume Training

The training script supports **robust checkpointing and resume logic**.

### Resume Training

To resume training from a checkpoint, set in `config.py`:

```python
"preload": "epoch_no" or "best"
```

For eg: To resume from epoch 29

```python
"preload": "29"
```

When resuming:

* Model weights are restored
* Optimizer state is restored
* Learning rate scheduler state is restored
* Global training step is restored
* Best BLEU score so far is restored

This allows training to continue seamlessly without losing learning dynamics.

### Saving Checkpoints

* **Save every N epochs**

```python
save_every = N
```

* **Save only the best BLEU checkpoint**

```python
save_best_only = True
```

If `save_best_only` is enabled, only checkpoints that improve BLEU are written.

If `save_every` is `None` and `save_best_only` is `False`, a checkpoint is saved at the end of every epoch.

---

## Monitoring Training

TensorBoard logs are written automatically.

To view them:

```bash
tensorboard --logdir runs
```

Tracked metrics:

* Training loss
* Validation BLEU (SacreBLEU)
* Validation WER
* Validation CER

---

## Pretrained Models

Pretrained weights for the **English → Italian** model (Opus Books) are available in the **[Releases](https://github.com/ankushhKapoor/transformer-from-scratch/releases/latest)** section of this repository.

### How to use:
1. Download the `model_best.pt` file from the latest release.
2. Place the file into the `weights/` directory.
3. Update `config.py` to point to the file:
   ```python
   "preload": "best"
   ```

**Educational Recommendation**

While pretrained weights are provided for convenience, this project is designed primarily as a learning resource. It is highly recommended to train your own models from scratch. Manually adjusting parameters in config.py and changing language subset is essential for observing how architectural choices directly impact the training and final performance of a Transformer.

---

## Decoding

The model supports:

* **Greedy decoding**
* **Beam search decoding with length penalty**

Beam search is used for BLEU evaluation, following standard NMT practice.

---

## Evaluation

Evaluation metrics:

* **SacreBLEU** (corpus-level)
* **Word Error Rate (WER)**
* **Character Error Rate (CER)**

BLEU is computed using `sacrebleu.corpus_bleu`, while WER and CER are computed using `torchmetrics`, ensuring standardized and reproducible evaluation.

---

## Attribution

This project was built with **guidance and conceptual help from [Umar Jamil’s Transformer tutorial](https://youtu.be/ISNdQcPhsts)**.

However:

* The code was written and structured independently
* Components were re-implemented to ensure full understanding
* The project was not created by copy-pasting tutorial code

The tutorial served as a learning reference, not a direct code source.

---