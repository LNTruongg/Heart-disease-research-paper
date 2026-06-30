# Heart Disease Risk Prediction using Machine Learning

This project builds a machine learning pipeline to **predict the risk of heart disease** from clinical data while providing comprehensive analyses of:
- Model performance
- Decision threshold optimization
- Probability calibration
- Logistic Regression interpretability

---

## 1. Problem Objective

The original dataset contains heart disease severity labels ranging from **0 to 4**.

In this notebook, the target is converted into a **binary classification** problem:

- **0** = No heart disease
- **1** = Heart disease (`num > 0`)

This transformation:
- Simplifies the prediction task
- Matches the objective of early disease detection
- Enables easier comparison among classical machine learning models

---

## 2. Dataset

The dataset used in this project is the **Cleveland Heart Disease Dataset**.

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

### Features Used

**Numerical Features**
- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

**Categorical Features**
- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

---

## 3. Data Preprocessing

The notebook includes data inspection, exploratory data analysis (EDA), and a complete preprocessing pipeline for model training.

### 3.1 Data Cleaning
- Load the CSV dataset
- Replace `"?"` with `NaN`
- Analyze missing values
- Create a temporary cleaned dataset for EDA

### 3.2 Train/Test Split
- 80/20 train-test split
- `stratify=y` to preserve class distribution

### 3.3 Preprocessing Pipeline

**Numerical Pipeline**
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

**Categorical Pipeline**
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

The preprocessing pipeline helps:
- Prevent data leakage
- Handle missing values properly
- Normalize numerical features
- Automatically encode categorical variables

---

## 4. Models Compared

The notebook evaluates five machine learning models:

1. Majority Baseline
2. Logistic Regression (L2 Regularization)
3. Logistic Regression (L1 Regularization)
4. Linear SVM
5. RBF SVM

---

## 5. Notebook Contents

The notebook is organized similarly to a research paper and includes the following sections.

### 5.1 Exploratory Data Analysis (EDA)

- Missing value analysis (before and after cleaning)
- Class distribution
- Boxplots
- Histograms
- KDE plots
- Count plots for categorical variables
- Correlation heatmap
- Feature distributions by target class
- Pairplot

### 5.2 Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (Average Precision)

### 5.3 Threshold Analysis

The notebook investigates how different decision thresholds affect:

- Precision
- Recall
- F1-score

Additional analyses include:

- Threshold tuning table (`0.3`, `0.4`, `0.5`, `0.6`, `0.7`)
- Confusion matrix at different thresholds
- Cost-based threshold selection for medical diagnosis

### 5.4 Probability Calibration

Beyond the calibration curve, the notebook also computes:

- Brier Score
- Log Loss
- Expected Calibration Error (ECE)

Calibration methods include:

- Reliability table (per probability bin)
- Platt Scaling
- Isotonic Regression

### 5.5 Model Interpretability

For Logistic Regression, the notebook extracts:

- Feature coefficients
- Positive and negative coefficient signs
- Odds ratios using `exp(coef)`

These analyses help explain which clinical variables increase or decrease the predicted risk of heart disease.

### 5.6 Reproducibility

To improve reproducibility, the notebook includes:

- Bootstrap 95% confidence intervals for ROC-AUC
- Export of evaluation results to Excel
- Reproducibility checklist

---

## 6. Results

### 6.1 Test Performance

| Model | Accuracy | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| LogReg_L2 | 0.885246 | 0.966450 | 0.963435 | 0.881356 | 0.838710 | 0.928571 |
| SVC_RBF | 0.885246 | 0.964286 | 0.955836 | 0.881356 | 0.838710 | 0.928571 |
| SVC_Linear | 0.852459 | 0.961039 | 0.957194 | 0.852459 | 0.787879 | 0.928571 |
| LogReg_L1 | 0.868852 | 0.958874 | 0.954618 | 0.862069 | 0.833333 | 0.892857 |
| Majority | 0.540984 | 0.500000 | 0.459016 | 0.000000 | 0.000000 | 0.000000 |

### 6.2 Best Performing Model

According to the experimental results, **Logistic Regression with L2 regularization** achieved the best overall performance.

- **ROC-AUC:** 0.966450
- **PR-AUC:** 0.963435
- **F1-score:** 0.881356
- **Recall:** 0.928571

These results indicate that Logistic Regression L2 provides excellent predictive performance while maintaining strong interpretability, making it a suitable model for clinical decision support.

### 6.3 Threshold Tuning

The notebook includes a threshold tuning analysis for the **Logistic Regression L2** model.

Different probability thresholds (`0.3`, `0.4`, `0.5`, `0.6`, and `0.7`) are evaluated to examine the trade-offs between precision and recall. For each threshold, the notebook reports:

- Precision
- Recall
- F1-score
- Confusion Matrix

A simple cost function is also applied to identify an operating threshold that is more appropriate for medical diagnosis, where minimizing false negatives is often more important than maximizing overall accuracy.

| Threshold | Precision | Recall | F1 | FP | FN | Cost |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.729730 | 0.964286 | 0.830769 | 10 | 1 | 15 |
| 0.4 | 0.818182 | 0.964286 | 0.885246 | 6 | 1 | 11 |
| 0.5 | 0.838710 | 0.928571 | 0.881356 | 5 | 2 | 15 |
| 0.6 | 0.862069 | 0.892857 | 0.877193 | 4 | 3 | 19 |
| 0.7 | 0.923077 | 0.857143 | 0.888889 | 2 | 4 | 22 |

### Threshold Analysis Summary

- Lower thresholds lead to higher recall.
- Higher thresholds lead to higher precision.
- A threshold of **0.4** achieves the lowest cost in the current evaluation table.
- A threshold of **0.5** provides a good balance between precision and recall.


### 6.4. Calibration metrics
| Model | Brier Score | Log Loss | ECE |
|---|---:|---:|---:|
| LogReg_L2 | 0.079737 | 0.259499 | 0.082536 |
| LogReg_L1 | 0.083293 | 0.272178 | 0.097331 |
| SVC_RBF | 0.086401 | 0.299525 | 0.108663 |
| SVC_Linear | 0.089112 | 0.298365 | 0.108011 |
| Majority | 0.459016 | 16.544628 | 0.459016 |

The results show that `LogReg_L2` is not only strong in classification performance but also achieves the best calibration among the main models in the study.


### 6.5. Bootstrap 95% CI for AUC
| Model | AUC Bootstrap Mean | AUC 95% CI Lower | AUC 95% CI Upper |
|---|---:|---:|---:|
| LogReg_L2 | 0.966139 | 0.923077 | 0.994444 |
| SVC_RBF | 0.963848 | 0.912902 | 0.995699 |
| SVC_Linear | 0.960644 | 0.911762 | 0.992376 |
| LogReg_L1 | 0.958471 | 0.910981 | 0.989515 |
| Majority | 0.500000 | 0.500000 | 0.500000 |

---

## 7. Overall pipeline structure

```text
Raw Data
   │
   ├── Replace '?' -> NaN
   │
   ├── Target Transformation
   │      num = 0   -> y = 0
   │      num > 0   -> y = 1
   │
   ├── Preprocessing
   │      ├── Numeric   -> Median Imputation -> StandardScaler
   │      └── Categorical -> Most Frequent Imputation -> OneHotEncoder
   │
   ├── Models
   │      ├── Majority Class
   │      ├── Logistic Regression (L1 / L2)
   │      └── SVC (Linear / RBF)
   │
   ├── Research Question 1
   │      Compare ROC-AUC / PR-AUC / F1
   │
   ├── Research Question 2
   │      Threshold trade-off
   │
   ├── Research Question 3
   │      Calibration check
   │
   └── Final Recommendation
```

## 8. How to Run the Notebook

### Step 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

### Step 2. Prepare the dataset
- Place the dataset file as `data.csv`
- Or update the `DATA_PATH` variable inside the notebook

### Step 3. Run the notebook
Open Jupyter Notebook or VS Code and run the cells sequentially.

---

## 9. Notebook Outputs

After running the notebook, the following outputs may be generated:
- EDA visualizations
- ROC curve
- Precision-Recall curve
- Calibration plot
- Confusion matrix
- Model evaluation table
- Threshold tuning table
- Calibration metrics table
- Bootstrap confidence interval table
- Aggregated results exported to Excel
- Image folder such as `figures/`

---

## 10. Practical Applications

This project can serve as a foundation for:
- Heart disease risk screening support systems
- Educational and research tool for machine learning in healthcare
- Demo model for health prediction dashboards
- Report or paper comparing classical machine learning models

**Note:** This is a machine learning model for educational and research purposes only, and it does **not replace real medical diagnosis**.

---

## 11. Strengths of the Notebook

- Complete pipeline from EDA to modeling
- Includes baseline comparison
- Includes threshold analysis
- Includes calibration analysis
- Includes interpretability analysis
- Includes bootstrap confidence intervals
- Ensures reproducibility and result export
- Suitable for reports, slides, and presentations

---

## 12. Suggested Improvements

To further enhance the project, you can add:
- GridSearchCV / Optuna for hyperparameter tuning
- SHAP for advanced interpretability
- External validation on another dataset
- Web app for user input and prediction
- Model saving using `joblib`
- Deployment using Streamlit or Flask

---

## 13. Main File

- Main Notebook: **`[Topic_11]_HEART_DISEASE_Nhom5_LeNhatTruong_TranThanhPhu.ipynb`**

---

## 14. Short Summary

This is a complete machine learning notebook for **heart disease prediction**, focusing on:
- Logistic Regression
- SVC
- Threshold optimization
- Probability calibration
- Model interpretability
- Reproducibility

The best performing model in this notebook is **Logistic Regression L2**.
