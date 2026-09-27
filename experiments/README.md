# Experiments Catalog & Directory Structure

This directory houses the complete iterative development history of the Human Activity Recognition (HAR) project, preserving all experimental paths, ablation trials, and rejected hypotheses alongside winning techniques.

---

## Directory Organization

```
experiments/
├── decision_tree/
│   └── decision_tree.py                # Initial Decision Tree baseline with EDA and preprocessing
│
├── svm/
│   ├── svm.py                          # Support Vector Machine (RBF) tuning and evaluation
│   └── submission1.py                  # Working baseline pipeline generating SVM submission
│
├── baselines/
│   ├── xgboost_model.py                # Gradient boosted decision trees (raw vs. per-subject normalized)
│   └── comparison.py                   # Side-by-side evaluation of Decision Tree, SVM, and Feedforward NN
│
├── preprocessing/
│   ├── explore.py                      # Exploratory data analysis, distributions, and missing value checks
│   ├── verification.py                 # Data integrity verification and feature structure assertions
│   ├── verify_rawalign.py              # Cross-verification of UCI HAR raw signals against Kaggle train.csv
│   ├── sensor_compare.py               # Ablation: Accelerometer-only vs. Gyroscope-only feature sets
│   ├── PCA_classification.py           # Dimensionality reduction: PCA vs. raw 561 features
│   ├── outlier.py                      # Transition window & outlier removal test within subject-activity groups
│   ├── sub_norm.py                     # Per-subject normalization experiment (diagnosing Subject 14 failure)
│   └── sub_training.py                 # Full 5-fold GroupKFold validation of per-subject vs. global normalization
│
├── neural_networks/
│   ├── feedforward.py                  # Staged hyperparameter sweep (depth, width, activations, dropout)
│   ├── improve.py                      # Regularization and architecture refinements for Feedforward MLP
│   ├── best_hyperpara.py               # Training Feedforward NN with confirmed optimal hyperparameters
│   ├── best_epoch.py                   # Early stopping investigation and epoch scheduling
│   ├── error_ckeck.py                  # Error distribution checks across validation splits
│   ├── Final_model.py                  # Evaluation script computing confusion matrices and per-class reports
│   └── Final_Sub.py                    # Standalone Feedforward NN submission generation pipeline
│
├── raw_signals_deep_learning/
│   ├── Raw_NN.py                       # Recurrent Neural Network (LSTM/GRU) on raw 128x9 time-series signals
│   ├── RNN_checks.py                   # Multi-seed stability and GroupKFold cross-validation for RNN
│   ├── rnn_persub.py                   # Per-channel per-subject normalization for raw sensor windows
│   ├── RNN_sub.py                      # Standalone RNN submission generation pipeline
│   ├── onedCNN.py                      # 1D Convolutional Neural Network on raw signals (5-seed ensemble)
│   └── multi_cnn.py                    # Multi-scale CNN with attention mechanisms (inspired by DCAM-Net)
│
├── ensemble/
│   ├── improved1.py                    # 5-seed Feedforward NN ensemble on full training dataset
│   ├── seedtest.py                     # Ensemble scaling: 5-seed vs. 10-seed diminishing returns check
│   ├── groupfold_en.py                 # GroupKFold cross-subject ensemble with early stopping
│   ├── persub_model.py                 # Per-subject normalization + 5-seed ensemble submission pipeline
│   ├── adabn.py                        # Adaptive Batch Normalization (AdaBN) test-time domain adaptation
│   ├── adabn_sub.py                    # Full pipeline: Per-subject norm + 5-seed ensemble + AdaBN
│   ├── blended_model.py                # Probability blend: Neural Network ensemble + SVM (RBF)
│   ├── persub_blended.py               # Multi-modal blend: Per-subject NN + Per-subject RNN via GroupKFold
│   └── rnn_fixedalpha.py               # GroupKFold blend evaluation with fixed alpha weighting
│
└── diagnostics_and_leakage/
    ├── splittest.py                    # Split robustness check: variance across different validation subjects
    ├── test_verify.py                  # Deep error analysis on Fold 0 (Subject 14/15/19/25 failure modes)
    ├── without_group.py                # Subject leakage experiment: including subject ID as an explicit input
    ├── Diff_check.py                   # Auditing row-level prediction differences between submission candidates
    └── sub_check.py                    # Formatting and schema assertion check for submission files
```

---

## Mapping of Advanced Techniques to the Final Champion Notebook

Several advanced techniques, ablations, and rejected architectures were engineered and executed directly inside the protected champion notebook [`FINAL_WON_SUB.ipynb`](../FINAL_WON_SUB.ipynb):

| Technique / Experiment | Location in `FINAL_WON_SUB.ipynb` | Description & Outcome |
| :--- | :--- | :--- |
| **Defensible Physics Features** | Cell 3 (`engineer_defensible_features`) | Added 8 physics-based interaction features (magnitude ratios, dot products, cos angles, jerk ratios). Dimension: 561 -> 569. |
| **ResMLP Architecture** | Cell 6 (`class ResMLP`, `class ResBlock`) | Residual MLP blocks with Layer/BatchNorm, GELU, and Dropout. Configurations: 512-dim (2 blocks) and 768-dim (3 blocks). |
| **DeepMLP Architecture** | Cell 6 (`class DeepMLP`, `Swish`) | Deep architectures (512-256-128, 1024-512-256, 512-128) using GELU and Swish activations with heavy dropout (0.38 - 0.45). |
| **Data Augmentation & Regularization** | Cell 6 (`train_nn_model`) | Input Gaussian noise ($\sigma = 0.008$), MixUp ($\alpha = 0.15$, $p = 0.4$), Label Smoothing ($\epsilon = 0.04$), and Class Weighting. |
| **15-Model Ensemble** | Cell 6 (`MODEL_CONFIGS`, `SEEDS`) | 5 distinct architectures $\times$ 3 random seeds (42, 43, 44), evaluated with AdaBN per subject. |
| **Pseudo-Labeling / Self-Training** | Cell 7 (`[ABLATION B]`) | **Rejected (0.9818 vs 0.9835 baseline)**: High-confidence pseudo-labels failed to improve held-out validation macro F1. |
| **Temporal Sequence Smoothing** | Cell 7 (`[ABLATION C]`, `temporal_smooth`) | **Adopted (0.9865 vs 0.9835 baseline)**: Audited test.csv subject contiguity (confirmed contiguous) and applied 3-tap kernel [0.15, 0.70, 0.15]. |
| **Hierarchical Cascade Classifier** | Cell 7.5 (`STAGE 2.5: HIERARCHICAL CLASSIFIER`) | **Rejected (0.9278 vs 0.9865 flat pipeline)**: 4-way coarse SVM + 2 binary decision trees (Sitting vs Standing, Upstairs vs Downstairs). Kept as documented negative result. |
