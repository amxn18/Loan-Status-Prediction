# Loan-Status-Prediction
cat > README.md << 'EOF'
# 🏦 Loan Status Prediction using SVM

This machine learning project predicts whether a loan will be approved based on applicant details using a Support Vector Machine (SVM) classifier.

---

## 📂 Dataset

The dataset used is `loanData.csv` which includes the following features:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (Target)

---

## 🛠️ Features & Preprocessing

- **Missing Values Handling:**
  - Categorical: Filled with **mode**
  - Numerical: Filled with **median**
  
- **Encoding:**
  - Label Encoding for categorical variables

- **Scaling:**
  - StandardScaler is used to standardize numerical features

- **Feature Transformation Example:**
  - 'Male' ➝ 1, 'Female' ➝ 0
  - 'Urban' ➝ 2, 'Semiurban' ➝ 1, 'Rural' ➝ 0

---

## 📊 Visualizations

- Education vs Loan Approval
- Marital Status vs Loan Approval

---

## 🧠 Model

- **Algorithm**: Support Vector Machine (Linear Kernel)
- **Train/Test Split**: 80/20
- **Evaluation**:
  - Accuracy on Training Set
  - Accuracy on Test Set

---

## 🔍 Prediction

You can pass a new input like:

```python
input_data = ['Male', 'No', 0, 'Graduate', 'No', 5000, 2000, 150, 360, 1, 'Urban']
