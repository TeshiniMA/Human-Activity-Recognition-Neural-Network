# Model Architectures & Ablation Study Reference

This document provides complete technical specifications for every machine learning and deep learning architecture evaluated in this project, followed by the rigorous ablation study that determined the champion configuration.

---

## 1. Classical Machine Learning Baselines

### 1.1 Decision Tree Classifier
- **Implementation**: `sklearn.tree.DecisionTreeClassifier`
- **Objective**: Establish an interpretable, non-parametric tree baseline.
- **Tuning Protocol**: 3-fold `GroupKFold` cross-validation on the training fold (grouped by `subject`).
- **Hyperparameter Grid**:
  - `max_depth`: `[8, 12, 16, 20, None]`
  - `min_samples_leaf`: `[1, 3, 5, 10]`
- **Best Hyperparameters**: `max_depth = 8`, `min_samples_leaf = 10`
- **Held-Out Validation Score**: Macro F1 = **0.9264**
- **Per-Class Breakdown**:
  - `LAYING`: 1.0000
  - `SITTING`: 0.9744
  - `STANDING`: 0.9752
  - `WALKING`: 0.8852
  - `WALKING_DOWNSTAIRS`: 0.8827
  - `WALKING_UPSTAIRS`: 0.8406

### 1.2 Support Vector Machine (RBF Kernel)
- **Implementation**: `sklearn.svm.SVC`
- **Objective**: Kernelized maximum-margin baseline suited for high-dimensional continuous features.
- **Tuning Protocol**: 3-fold `GroupKFold` cross-validation on training fold.
- **Hyperparameter Grid**:
  - `C`: `[1.0, 10.0, 50.0]`
  - `gamma`: `['scale', 0.01, 0.001]`
- **Best Hyperparameters**: `C = 50.0`, `gamma = 0.001`
- **Held-Out Validation Score**: Macro F1 = **0.9625**
- **Per-Class Breakdown**:
  - `LAYING`: 0.9784
  - `SITTING`: 0.9483
  - `STANDING`: 0.9552
  - `WALKING`: 0.9825
  - `WALKING_DOWNSTAIRS`: 0.9336
  - `WALKING_UPSTAIRS`: 0.9768

### 1.3 XGBoost (Gradient Boosted Decision Trees)
- **Implementation**: `xgboost.XGBClassifier`
- **Objective**: Determine whether sequentially-corrected gradient boosting trees outperform single trees and rival neural approaches.
- **Evaluated**: Raw vs. per-subject normalized features across GroupKFold splits.

---

## 2. Neural Network Architectures

### 2.1 Feedforward Neural Network (Early Iteration)
- **Architecture**:
  - Input Layer: 561 dimensions
  - Hidden Layer 1: 256 units + `BatchNorm1d` + `ReLU` + `Dropout(0.5)`
  - Hidden Layer 2: 128 units + `BatchNorm1d` + `ReLU` + `Dropout(0.5)`
  - Output Layer: 6 units (Linear logits)
- **Optimizer**: Adam ($lr = 10^{-3}$, weight decay = 0.0)

### 2.2 ResMLP (Residual Multi-Layer Perceptron)
- **Motivation**: Deep feedforward networks suffer from vanishing gradients and degradation. Skip connections enable deeper feature extraction without optimization bottlenecks.
- **Building Block (`ResBlock`)**:
  $$\mathbf{h}_1 = \text{GELU}(\text{BatchNorm}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1))$$
  $$\mathbf{h}_2 = \text{Dropout}(\mathbf{h}_1, p)$$
  $$\mathbf{h}_3 = \text{BatchNorm}(\mathbf{W}_2 \mathbf{h}_2 + \mathbf{b}_2)$$
  $$\mathbf{y} = \text{GELU}(\mathbf{x} + \mathbf{h}_3)$$
- **Variants in Final Ensemble**:
  - `ResMLP_512`: Hidden dim = 512, 2 Residual Blocks, Dropout = 0.35
  - `ResMLP_768`: Hidden dim = 768, 3 Residual Blocks, Dropout = 0.30

### 2.3 DeepMLP with Swish and GELU Activations
- **Motivation**: Provide architectural diversity to the ensemble using smooth non-monotonic activations (Swish: $x \cdot \sigma(x)$ and GELU: $x \Phi(x)$) capable of capturing subtle non-linearities.
- **Variants in Final Ensemble**:
  - `512_256_128`: Hidden `[512, 256, 128]`, Dropout = 0.40, Activation = GELU
  - `1024_512_256`: Hidden `[1024, 512, 256]`, Dropout = 0.45, Activation = Swish
  - `512_128`: Hidden `[512, 128]`, Dropout = 0.38, Activation = Swish

### 2.4 Raw Time-Series Models (Evaluated in Phase 4)
- **Recurrent Neural Networks (RNN / GRU)**:
  - Input: $128 \text{ timesteps} \times 9 \text{ raw sensor channels}$
  - Hidden State: 64/128 GRU units, bidirectional
- **1D Convolutional Neural Network (1D-CNN)**:
  - 3 sequential 1D Convolution blocks (Conv1D + BatchNorm + ReLU + MaxPool1D) + Fully Connected classification head
- **Multi-Scale CNN with Channel Attention**:
  - Multi-branch temporal receptive fields (kernel sizes 3, 5, 7) combined with Squeeze-and-Excitation channel attention.

---

## 3. Regularization & Data Augmentation Techniques

In the champion pipeline (`FINAL_WON_SUB.ipynb`), models are trained with a composite regularization protocol:

1. **Input Gaussian Jitter**:
   $$\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, 0.008^2 \mathbf{I})$$
   Prevents the model from overfitting to exact sensor quantization boundaries.
2. **Manifold MixUp ($p = 0.4, \alpha = 0.15$)**:
   $$\lambda \sim \text{Beta}(\alpha, \alpha)$$
   $$\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1 - \lambda) \mathbf{x}_j, \quad \mathcal{L} = \lambda \mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}_i) + (1 - \lambda) \mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}_j)$$
   Smooths inter-class decision boundaries, mitigating overconfident misclassifications.
3. **Label Smoothing ($\epsilon = 0.04$)**:
   Prevents softmax saturation on ambiguous borderline samples.
4. **Cosine Annealing Learning Rate**:
   Decays learning rate smoothly from $10^{-3}$ to $10^{-5}$ across training epochs.
5. **Class-Weighted Loss**:
   Inverse class frequency weighting ensures minority activity classes receive balanced gradient signal.

---

## 4. Test-Time Domain Adaptation (AdaBN)

- **Reference**: Li et al., *"Revisiting Batch Normalization for Practical Domain Adaptation"* (2016).
- **Mechanism**: At prediction time, for each unseen subject $s$ in the test set, we freeze all learned network weights $\mathbf{W}, \mathbf{b}, \boldsymbol{\gamma}, \boldsymbol{\beta}$. We set `BatchNorm` momentum to 1.0 and run a single forward pass over all unlabeled samples from subject $s$ in `train` mode.
- **Effect**: This directly replaces the population running statistics $(\mu_{\text{pop}}, \sigma^2_{\text{pop}})$ with subject-specific activation statistics $(\mu_s, \sigma^2_s)$. The network is thus dynamically calibrated to each individual's physical telemetry before computing final softmax probabilities.

---

## 5. Controlled Ablation Study

Every ablation was evaluated on the identical held-out validation split derived via `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` on unseen subjects ([1, 3, 15, 25, 27]).

### Summary Table

| Stage / Component | Macro F1 | Delta vs. Base Ensemble | Status | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Decision Tree (Tuned)** | 0.9264 | -0.0571 | Baseline | Single-tree benchmark |
| **SVM RBF (Tuned)** | 0.9625 | -0.0210 | Baseline | Kernel baseline |
| **A: Base NN Ensemble (15 Models)** | **0.9835** | **0.0000** | Candidate | 5 architectures $\times$ 3 seeds with AdaBN |
| **B: + Pseudo-Labeling (Self-Training)** | **0.9818** | **-0.0017** | **REJECTED ❌** | Did not improve validation score |
| **C: + Temporal Probability Smoothing** | **0.9865** | **+0.0030** | **ADOPTED ✅** | Improved F1 & confirmed contiguous |
| **D: Combined Production Pipeline** | **0.9865** | **+0.0030** | **CHAMPION 🏆** | Ensemble + AdaBN + Temporal Smoothing |
| **Hierarchical Cascade Classifier** | **0.9278** | **-0.0557** | **REJECTED ❌** | Error propagation in sub-classifiers |

---

## 6. Deep Dive into Rejected Approaches

### 6.1 Why Pseudo-Labeling Failed (-0.0017)
- **Mechanism**: The base ensemble predicted probabilities on the unlabeled validation pool. Samples with maximum predicted confidence $\ge 0.95$ (1,335 samples) were assigned hard pseudo-labels and appended to the training set for a second retraining round.
- **Failure Analysis**: In human activity recognition, the hardest errors occur on ambiguous transition windows and boundary postures (e.g., dynamic sitting transitions). Highly confident pseudo-labels on slightly misaligned static postures caused confirmation bias—reinforcing false confidence on borderline samples and deteriorating out-of-fold generalization from 0.9835 to 0.9818.

### 6.2 Why the Hierarchical Cascade Failed (-0.0557)
- **Mechanism**: Designed a two-stage cascade targeting the two known hard pairs:
  - Stage 1: Coarse 4-group SVM (`STATIC_SIT_STAND`, `LAYING`, `WALK_UP_DOWN`, `WALKING`).
  - Stage 2A: Dedicated binary Decision Tree for `SITTING` vs. `STANDING`.
  - Stage 2B: Dedicated binary Decision Tree for `WALKING_UPSTAIRS` vs. `WALKING_DOWNSTAIRS`.
- **Failure Analysis**:
  - Stage 1 performed strongly (Coarse Macro F1 = 0.9872).
  - However, Stage 2A binary separation achieved only **0.8639** macro F1 within static rows, and Stage 2B achieved **0.9516**.
  - Any error committed in Stage 1 was fatal (unrecoverable by sub-classifiers), and the specialized sub-classifiers lacked the rich multi-task regularizing context present in the flat 15-model neural ensemble.
  - Overall cascade score dropped to **0.9278**, significantly worse than the flat pipeline's **0.9865**.

---

## 7. Verification of Temporal Sequence Probability Smoothing

### Mechanism
Human physical activities possess high temporal autocorrelation; a subject does not typically switch between walking and sitting for a single 2.56-second window. A 3-tap normalized smoothing kernel was defined:
$$\mathbf{p}_t^{\text{smoothed}} = 0.15 \cdot \mathbf{p}_{t-1} + 0.70 \cdot \mathbf{p}_t + 0.15 \cdot \mathbf{p}_{t+1}$$

### Essential Pre-Condition: Test Row Contiguity Audit
Applying a sequential filter to shuffled or non-contiguous data would average across completely unrelated activities, corrupting predictions. Before adopting temporal smoothing, an automated contiguity audit was executed directly on `test.csv`:
- Subject 2: 302 samples — Contiguous: `True`
- Subject 4: 317 samples — Contiguous: `True`
- Subject 9: 288 samples — Contiguous: `True`
- Subject 10: 294 samples — Contiguous: `True`
- Subject 12: 320 samples — Contiguous: `True`
- Subject 13: 327 samples — Contiguous: `True`
- Subject 18: 364 samples — Contiguous: `True`
- Subject 20: 354 samples — Contiguous: `True`
- Subject 24: 381 samples — Contiguous: `True`
- **Result**: `test.csv` rows are 100% contiguous per subject. Temporal smoothing is mathematically justified, verified safe, and boosted Macro F1 to **0.9865**.
