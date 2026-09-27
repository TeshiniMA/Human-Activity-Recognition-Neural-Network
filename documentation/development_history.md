# Project Development History & Experimental Narrative

This document chronicles the complete research and engineering trajectory of the CO5420 Human Activity Recognition (HAR) project, documenting how rigorous error diagnosis and validation-guided iterations transformed early baseline models into the final champion solution.

---

## 1. Project Inception & Problem Formulation

Human Activity Recognition via smartphone sensor telemetry represents a complex classification challenge with 6 physical activities:
1. `LAYING` (Static)
2. `SITTING` (Static)
3. `STANDING` (Static)
4. `WALKING` (Dynamic)
5. `WALKING_UPSTAIRS` (Dynamic)
6. `WALKING_DOWNSTAIRS` (Dynamic)

The raw sensor dataset captures triaxial linear acceleration and triaxial angular velocity from a Samsung Galaxy S II worn on the waist, sampled at 50 Hz. From these raw continuous signals, 561 hand-crafted statistical and frequency-domain features were extracted over 2.56-second sliding windows (128 readings, 50% overlap).

---

## 2. Phase 1: Classical Baselines & Dimensionality Reduction

### Experimental Hypothesis
Establish baseline benchmarks on the hand-crafted 561-feature dataset using standard classical classifiers (Decision Tree, Support Vector Machine, and XGBoost) and evaluate whether Principal Component Analysis (PCA) improves generalization by removing redundant collinear dimensions.

### Scripts
- `experiments/decision_tree/decision_tree.py`
- `experiments/svm/svm.py`
- `experiments/svm/submission1.py`
- `experiments/baselines/xgboost_model.py`
- `experiments/preprocessing/PCA_classification.py`
- `experiments/baselines/comparison.py`

### Findings & Empirical Results
- **Decision Tree**: Rapidly overfit deeper trees. When constrained via grid search, it achieved a validation macro F1 of **0.8697**. Per-class inspection revealed significant confusion between `SITTING` and `STANDING` (F1 0.8852 and 0.8768) and between `WALKING_UPSTAIRS` and `WALKING_DOWNSTAIRS` (F1 0.8068 and 0.7963).
- **Support Vector Machine (RBF Kernel)**: Significantly outperformed Decision Trees, achieving **0.9581** validation macro F1 with $C=50.0, \gamma=0.001$. SVM achieved perfect separation on `LAYING` (F1 = 1.0000).
- **PCA Experiment**: Reducing features to 95% variance (retaining ~100 principal components) consistently degraded performance:
  - Decision Tree: 0.87 (raw) vs. 0.82 (PCA)
  - SVM: 0.96 (raw) vs. 0.95 (PCA)
- **Decision**: PCA was discarded. Subsequent modeling retained all raw sensor features.

---

## 3. Phase 2: Sensor Modality Contribution Study

### Experimental Hypothesis
Determine the relative discriminative power and complementarity of the Accelerometer (345 features) versus the Gyroscope (213 features).

### Script
- `experiments/preprocessing/sensor_compare.py`

### Findings & Empirical Results
- **All Features (561)**: Macro F1 = **0.9581**
- **Accelerometer-Only (345)**: Macro F1 = **0.9478**
- **Gyroscope-Only (213)**: Macro F1 = **0.8143**

#### Per-Class Insights
- Accelerometer features are indispensable for static posture detection (`LAYING` F1: 0.9985 for Acc-only vs. 0.6746 for Gyro-only). This aligns with physics: gravity acceleration vector alignment defines body orientation.
- Gyroscope features alone struggle with static orientations, but contribute substantially to distinguishing walking up versus down stairs.
- **Decision**: Both sensor modalities are complementary and required for optimal classification.

---

## 4. Phase 3: Feedforward Neural Network Architecture Search

### Experimental Hypothesis
Construct a PyTorch Multilayer Perceptron (MLP) operating directly on the 561 features to surpass the SVM baseline.

### Scripts
- `experiments/neural_networks/feedforward.py`
- `experiments/neural_networks/improve.py`
- `experiments/neural_networks/best_hyperpara.py`
- `experiments/neural_networks/best_epoch.py`
- `experiments/neural_networks/Final_model.py`

### Findings & Empirical Results
- A staged architecture sweep evaluated depth (2 vs. 3 layers), width (128 to 512 units), activation functions (ReLU vs. GELU), Dropout rates (0.2 to 0.6), and Batch Normalization.
- Optimal architecture: `[561 -> 256 -> 128 -> 6]` with Batch Normalization, Dropout = 0.5, and Adam optimizer ($lr = 10^{-3}$, weight decay = 0).
- The Feedforward NN achieved a single-split validation macro F1 of **0.9765**, beating SVM (**0.9581**) across nearly every class.

---

## 5. Phase 4: End-to-End Deep Learning on Raw Sensor Signals

### Experimental Hypothesis
Rather than relying on 561 engineered features, can sequence models (RNN/GRU, 1D-CNN, Multi-Scale CNN) operating directly on the raw continuous time-series ($128 \times 9$ channels) extract superior temporal patterns?

### Scripts
- `experiments/raw_signals_deep_learning/Raw_NN.py`
- `experiments/raw_signals_deep_learning/RNN_checks.py`
- `experiments/raw_signals_deep_learning/rnn_persub.py`
- `experiments/raw_signals_deep_learning/onedCNN.py`
- `experiments/raw_signals_deep_learning/multi_cnn.py`

### Findings & Empirical Results
- On the initial single 80/20 train/val split, a GRU model achieved an extraordinary validation score of **0.9930**.
- **Crucial Diagnostic Check**: When subjected to a 5-fold GroupKFold validation across different held-out subjects, the RNN score plummeted to a mean of **0.9343** (with standard deviation 0.0352, ranging from 0.8973 to 0.9887).
- Multi-scale CNN with attention (inspired by DCAM-Net) and 1D-CNN models exhibited similar sensitivity to subject variation.
- **Conclusion**: The raw signal models were heavily susceptible to subject-specific sensor calibration and body biomechanics, generalizing worse to unseen subjects than models trained on engineered features.

---

## 6. Phase 5: Split Sensitivity & The Subject 14 Failure Diagnosis

### Experimental Hypothesis
Investigate why local validation scores (0.9746+) consistently exceeded leaderboard test performance (~0.9504).

### Scripts
- `experiments/diagnostics_and_leakage/splittest.py`
- `experiments/diagnostics_and_leakage/test_verify.py`

### Findings & Empirical Results
- In `splittest.py`, running 5-fold GroupKFold on the Feedforward NN revealed substantial score variance:
  - Fold 0 (Subjects [14, 15, 19, 25]): **0.8856**
  - Fold 1 (Subjects [6, 16, 21, 22]): **0.9077**
  - Fold 2 (Subjects [3, 11, 17, 26]): **0.9880**
  - Fold 3 (Subjects [1, 7, 23, 30]): **0.9711**
  - Fold 4 (Subjects [5, 8, 27, 28, 29]): **0.9405**
  - **Mean: 0.9386 $\pm$ 0.0381**
- Deep error analysis on Fold 0 isolated the root cause:
  - **Subject 14 alone was responsible for 90 out of ~152 total validation errors!**
  - Subject 14 exhibited an atypical personal walking gait that, when globally scaled with other subjects, shifted `WALKING` and `WALKING_DOWNSTAIRS` features into the `WALKING_UPSTAIRS` cluster.

---

## 7. Phase 6: The Normalization Breakthrough — Per-Subject Normalization

### Experimental Hypothesis
Instead of fitting a global scaler across all subjects, normalize each subject's features using that subject's *own* mean and standard deviation:
$$\hat{x}^{(s)} = \frac{x^{(s)} - \mu^{(s)}}{\sigma^{(s)} + \epsilon}$$
Because this requires no activity labels, it operates identically at inference time on `test.csv` using the test subject identifiers.

### Scripts
- `experiments/preprocessing/sub_norm.py`
- `experiments/preprocessing/sub_training.py`
- `results/validation/processed_persubj_comparison.csv`

### Findings & Empirical Results
- On Fold 0, per-subject normalization immediately eliminated Subject 14's misclassifications, elevating the fold score from **0.8856 to 0.9806** (+0.0951 gain!).
- Verified across all 5 folds:
  | Fold | Global Normalization Macro F1 | Per-Subject Normalization Macro F1 | Delta |
  | :---: | :---: | :---: | :---: |
  | Fold 0 | 0.8856 | **0.9806** | +0.0950 |
  | Fold 1 | 0.9077 | **0.9409** | +0.0332 |
  | Fold 2 | 0.9880 | **0.9900** | +0.0020 |
  | Fold 3 | 0.9711 | **0.9854** | +0.0143 |
  | Fold 4 | 0.9405 | **0.9683** | +0.0278 |
  | **Mean** | **0.9386 $\pm$ 0.0381** | **0.9731 $\pm$ 0.0176** | **+0.0345** |
- Every single fold improved, and cross-fold variance dropped by more than 50%. This represented the decisive breakthrough of the project.

---

## 8. Phase 7: Ensembling & Test-Time Adaptation (AdaBN)

### Experimental Hypothesis
Evaluate whether multi-seed ensembling and Adaptive Batch Normalization (AdaBN) can further reduce prediction variance on test subjects.

### Scripts
- `experiments/ensemble/improved1.py`
- `experiments/ensemble/seedtest.py`
- `experiments/ensemble/adabn.py`
- `experiments/ensemble/adabn_sub.py`

### Findings & Empirical Results
- **Multi-Seed Ensembling**: Averaging softmax probabilities across 5 independently seeded models smoothed decision boundaries and raised test performance from 0.95039 to 0.95701. A 10-seed experiment (`seedtest.py`) showed diminishing returns beyond 5 seeds.
- **Adaptive Batch Normalization (AdaBN)**: Recalibrating internal BatchNorm running statistics on each test subject's unlabeled data prior to inference improved validation macro F1 on 4 out of 5 folds (+0.0024 mean).

---

## 9. Phase 8: The Final Champion Pipeline (`FINAL_WON_SUB.ipynb`)

All cumulative insights were unified into a production Kaggle notebook implementing:
1. **Defensible Physics Feature Engineering**: 8 new interaction features (561 -> 569).
2. **Per-Subject Normalization + RobustScaler**: Mitigating subject-level domain shift.
3. **15-Model Multi-Architecture Ensemble**:
   - 2 ResMLP architectures (512-dim 2 blocks, 768-dim 3 blocks)
   - 3 DeepMLP architectures (GELU, Swish activations, heavy dropout)
   - 3 random seeds (42, 43, 44) per configuration
4. **Regularization & Augmentation**:
   - Input noise ($\sigma = 0.008$)
   - MixUp ($\alpha = 0.15$, $p = 0.4$)
   - Label smoothing ($\epsilon = 0.04$)
   - Class-balanced loss weights
5. **Controlled Ablation Study**:
   - Base 15-Model Ensemble: **0.9835**
   - Pseudo-Labeling Simulation: **0.9818** (*Rejected*)
   - Temporal Smoothing (3-tap window on audited contiguous rows): **0.9865** (*Adopted*)
   - Hierarchical Cascade Classifier: **0.9278** (*Rejected*)
6. **Final Retraining**: Full train set retraining for 35 epochs with AdaBN and temporal smoothing yielding `submission_improved.csv`.
