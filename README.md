# SHL Hiring Assessment 2026 — Spoken Grammar Scoring Engine

A documented Kaggle notebook exploring regression models for estimating spoken-English grammar scores from audio. The project combines acoustic descriptors with transcript-derived linguistic and timing features.

> **Dataset notice:** the SHL/Kaggle competition data is not included. Obtain it only through the official competition while authorized to access it. Do not upload competition audio, labels, generated transcripts, feature caches, or submission files to this public repository unless the competition rules explicitly permit that sharing.

## Project contents

```text
shl-grammar-scoring-engine/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── notebooks/
    └── grammar_scoring.ipynb
```

## Approach

1. **Dataset and integrity checks** — inspect the provided train/test CSVs, available recordings, and consistency of identifiers.
2. **Acoustic features** — extract duration, energy/activity, zero-crossing, spectral, MFCC, delta, contrast, roll-off, flatness, and pitch-related statistics with `librosa`.
3. **Acoustic baselines and ensembles** — compare regression approaches including Ridge, Random Forest, ExtraTrees, HistGradientBoosting, and CatBoost; use five-fold out-of-fold predictions to compare blends.
4. **Distribution-shift diagnostics** — train a cross-validated classifier to estimate how distinguishable the training and test feature distributions are. Treat this as a diagnostic, not proof of causality.
5. **Speech-to-text** — transcribe the recordings with Faster-Whisper (`small.en`) using a CUDA GPU and save a resumable JSONL cache in the Kaggle working directory.
6. **Linguistic/timing features** — extract lexical diversity, word and sentence statistics, function-word rates, repetition/filler patterns, speaking-rate estimates, pauses and ASR confidence statistics.
7. **Multimodal ensemble** — combine the acoustic ensemble with a text ExtraTrees model and validate the blend on matching five-fold splits.
8. **Submission validation** — generate a CSV with `filename,label`, preserve `test.csv` ordering and check for missing or duplicate filenames.

## Recorded results

The following are the results recorded in the notebook; cross-validation metrics and the competition's public score are different measures and must not be compared as if they were the same metric.

| Experiment | Pearson (OOF) | RMSE (OOF) | MAE (OOF) |
|---|---:|---:|---:|
| Original acoustic ensemble | 0.7967 | 0.7490 | 0.5848 |
| Robust acoustic blend | 0.7956 | 0.7514 | 0.5898 |
| Text-only ExtraTrees | 0.7936 | 0.7571 | 0.5811 |
| 50% acoustic / 50% text (pooled OOF) | 0.8509 | 0.6804 | 0.5496 |

The separate mean-of-fold metrics for the 50% text blend were Pearson 0.8477, RMSE 0.6799, and MAE 0.5496; averages of fold metrics can differ from metrics computed over all pooled out-of-fold predictions.

**Public leaderboard snapshot:** the Kaggle submissions page displayed a score of **0.4912** for the multimodal submission, compared with **0.6874** and **0.6788** for the two earlier acoustic submissions. This repository does not infer the hidden qualification threshold from these values; interpret the numbers using the official competition's score definition.

## Dataset access and expected layout

The dataset is deliberately not bundled. If you are authorized to participate, attach/download the dataset from the official Kaggle competition. The notebook as provided is configured for the Kaggle path:

```text
/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/*.wav
└── test/*.wav
```

The feature-extraction and model cells use this path directly. To run outside Kaggle, adjust the `BASE`/`DATA_DIR` definitions to point to your authorized local copy and configure a compatible PyTorch runtime.

See [`data/README.md`](data/README.md) for details. No competition data, transcript cache, derived feature CSVs, model checkpoints or submission CSVs are part of this ZIP.

## Environment and execution

**Recommended:** run on Kaggle with the competition dataset attached, Internet enabled for downloading the Whisper model, and a CUDA-enabled GPU (the recorded run used a Tesla T4). The complete workflow transcribes 985 recordings and can take around 20 minutes for transcription alone on the recorded runtime; total runtime varies.

1. Create a Kaggle notebook and attach the official competition data.
2. Upload/open `notebooks/grammar_scoring.ipynb`.
3. Ensure the accelerator is a CUDA GPU and Internet access is enabled.
4. Run cells in order. The notebook pins `faster-whisper==1.2.1` and `av==18.1.0` in its Whisper setup cell to avoid the PyAV 19 API incompatibility encountered during this experiment.
5. The notebook writes intermediate cache/features and the final submission under `/kaggle/working/`. These generated artifacts are intentionally ignored by Git.

For an external environment, create a virtual environment, install compatible PyTorch/CUDA build instructions for the machine first if using GPU, then install the packages listed in `requirements.txt`. `torch` wheel selection depends on operating system and CUDA runtime, so a working Kaggle CUDA installation is the reference environment.

## Limitations and interpretation

- Whisper ASR can normalize or misrecognize speech, so transcripts are imperfect proxies for the original spoken grammar.
- Train/test feature distribution shift was observed; cross-validation performance may not fully predict public or private leaderboard performance.
- The blend weight was selected after comparing weights on the recorded out-of-fold predictions. That selection can make the reported best validation score optimistic; nested validation or a fixed, prior-selected weight would provide a stricter estimate.
- The public notebook intentionally omits row-level outputs and the sample test transcript. Aggregate results are retained.
- One earlier robust-feature experiment produced an in-sample domain ROC-AUC of 1.0000. That estimate was not valid as an out-of-sample measure; the public notebook updates this diagnostic to five-fold cross-validation.
- OOF metrics, the Kaggle public score, and any private SHL qualification threshold are distinct. No threshold claim is made here.

## Security and repository hygiene

Do not commit competition audio/CSV files, transcript JSONL files, derived feature tables, generated submission CSVs, `.env` files, access tokens, or other credentials. Review the competition rules and any SHL instructions before publicly sharing project code or derived artifacts. No open-source license is included; add one only after confirming that you have the right to license the material you publish.
