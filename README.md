# Deep-Learning-Project

# Parkinson's Disease Detection Using Deep Learning

This project implements deep learning models to detect Parkinson's Disease (PD) and estimate its severity using speech data. It leverages sequence models like LSTM, GRU, and SimpleRNN to classify individuals as healthy or affected by PD and predict their Unified Parkinson's Disease Rating Scale (UPDRS) scores.

---

## 🧠 Problem Statement

Parkinson's Disease affects speech characteristics, and this project aims to:
- **Classify** subjects as either **healthy** or having **PD**
- **Predict** the **UPDRS score**, which quantifies the severity of the disease

---

## 📊 Dataset

**Source:** [UCI Parkinson Speech Dataset](https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+Audio+recordings)

- 40 subjects: 20 with PD, 20 healthy
- 26 voice recordings per subject (vowels, numbers, words, sentences)
- 26 acoustic features extracted from each recording
- Test set includes 28 PD patients with repeated vowel sounds
- Each PD recording has a corresponding UPDRS score

---

## 🔄 Data Preprocessing

- **Loading Data:** Read `.txt` files using `pandas`
- **Column Naming:** Features labeled as `f1` to `f26`, with `class` and `UPDRS`
- **Type Conversion:** Proper conversion for `subject_id`, `class`, and `UPDRS`
- **Trimming:** Ensures row count is a multiple of 26 (samples per subject)
- **Feature Scaling:** StandardScaler applied to both train/test
- **Reshaping:** Converted to 3D arrays (samples grouped by subject)
  - Train: `(40, 26, 26)`
  - Test: `(6, 26, 26)`
- **Target Variables:**
  - `y_train_class`: One class label per subject
  - `y_train_updrs`, `y_test_updrs`: For regression

---

## 🧱 Model Architectures

Models are implemented using **TensorFlow/Keras**, each encapsulated in a function:
- **LSTM:** Bidirectional LSTM with LayerNorm, Dense, Dropout
- **GRU:** Bidirectional GRU with BatchNorm, Dense, Dropout
- **BiLSTM:** Similar to LSTM with variations in hyperparameters
- **SimpleRNN:** Multiple SimpleRNN layers with Dropout and BatchNorm
- **DenseNet1D:** 1D CNN inspired by DenseNet using Conv1D layers

---

## 🏋️ Model Training & Compilation

- Compiled using the **Adam** optimizer
- Loss:
  - **Classification:** `sparse_categorical_crossentropy`
  - **Regression:** `mean_squared_error`
- Metrics:
  - **Accuracy** for classification
  - **Mean Absolute Error (MAE)** for regression
- Training uses `train_test_split`, EarlyStopping, and ReduceLROnPlateau

---

## 📈 Results & Evaluation

Each model's performance is evaluated on both classification and regression tasks.

### 🔢 Classification Accuracy
- **SimpleRNN:** ⭐ **87.5%**

### 📉 Regression MAE (UPDRS Prediction)
- **GRU:** ⭐ **8.8119**

---

## 📊 Comparison & Discussion

- Sequential models outperform CNN-based DenseNet1D for classification.
- GRU excels in UPDRS regression, likely due to its efficient gating mechanism.
- This shows the importance of model architecture for task-specific performance.

---

## 📌 Conclusion

This project demonstrates the effectiveness of deep learning in:
- Diagnosing PD from voice samples
- Estimating disease severity (UPDRS)
- Highlighting the strength of RNN-based models for temporal medical data

---

## 🛠️ Tools & Technologies

- Python
- Pandas, NumPy
- TensorFlow / Keras
- Scikit-learn
- Matplotlib, Seaborn


