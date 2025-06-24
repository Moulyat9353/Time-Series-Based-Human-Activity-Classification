This repository contains the implementation of binary and multi-class classification models using **Logistic Regression, L1-penalized Logistic Regression, and Naive Bayes** on time-domain features extracted from human activity time series data. The project focuses on the effect of varying time series segmentation and feature selection on classification performance.  

# 🧠Problem Statement:  

The objective is to classify human physical activities (binary or multi-class) based on time-domain features extracted from multivariate time series. The dataset contains 6 time series signals per instance. We experiment with segmenting these time series into smaller chunks (varying l), extracting statistical features from each segment, and evaluating different classification strategies on the transformed feature sets.

Key challenges addressed:

- Optimal segmentation length (l)

- Feature selection via p-values and Recursive Feature Elimination (RFE)

- Regularization via L1-penalized Logistic Regression

- Dealing with class imbalance and instability in logistic regression

- Comparison of Gaussian and Multinomial Naive Bayes for multi-class classification

# 📂 Dataset:
- Each instance: 6 raw time series (e.g., accelerometer/gyroscope signals)

- Labels: Binary (e.g., class 0 vs. class 1) or Multi-class (e.g., multiple activities)

- Goal: Extract features from segments of time series and classify activities

# ⚙️ Methods and Experiments:
**1) Binary Classification:**  

**Logistic Regression (with and without RFE):**
- Time series split into l = {2, ..., 20} segments

- Extracted time-domain features per segment: mean, std, max, min, etc.
  
- Feature selection via:

  - p-values from logistic regression coefficients

  - Recursive Feature Elimination (RFE)

- Evaluation:

  - 5-fold Stratified Cross-Validation

  - Confusion matrix, ROC curves, AUC

  - omparison of CV error vs. test set error

**L1 Penalized Logistic Regression:**  
- Used sklearn’s LogisticRegression(penalty='l1', solver='liblinear')

- Cross-validated on:

  - l: number of segments

  - C: inverse regularization strength

- Compared performance and interpretability with p-value-based feature selection  

**2) Multi-class Classification:** 

**L1-Penalized Multinomial Logistic Regression**  

- Classify all activity types (realistic setting)

- Grid search over (l, C)

- Evaluation: Test error, confusion matrix, multi-class ROC curves

**Naive Bayes (Gaussian and Multinomial)**

- Applied to raw and PCA-transformed features

- Evaluated with confusion matrix and class-wise ROC

# 🛠️Technologies Used:  
- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib, seaborn
- imblearn for SMOTE
- statsmodels for p-value extraction

# 📌 Notes

- Time to recurrence is not used for classification.

- Stratified CV is used to preserve class ratios.

- Missing values are handled via median imputation.
