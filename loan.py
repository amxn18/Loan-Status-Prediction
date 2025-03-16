import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

loanData = pd.read_csv('loanData.csv')
pd.set_option('future.no_silent_downcasting', True)

# PreProcessing
# print(loanData.isnull().sum())

# PLOTTING TO CHECK DISTRIBUTION
# x = plt.subplots(figsize=(10, 10))
# sns.displot(loanData)
# plt.show()
# USING ABOVE 3 LINES WE CAN SEE DATA IS SKEWED SO WE WILL USE MEDIAN TO FILL THE MISSING VALUES

# AS THE DATA CONTAINS BOTH NUMERIACL AND TEXT VALUES WE WILL USE MODE FOR TEXT AND MEDIAN FOR NUMERICAL VALUES
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    loanData[col].fillna(loanData[col].mode()[0], inplace=True)


numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'ApplicantIncome', 'CoapplicantIncome']
for col in numerical_cols:
    loanData[col].fillna(loanData[col].median(), inplace=True)
# print(loanData.isnull().sum())

# ENCODING CATEGORICAL DATA
loanData['Loan_Status'] = loanData['Loan_Status'].replace({'N': 0, 'Y': 1})
# print(loanData.head())
loanData['Dependents'] = loanData['Dependents'].replace(to_replace='3+', value='4')
loanData['Dependents'] = loanData['Dependents'].astype(int)
# print(loanData['Dependents'].value_counts())

# DATA VISUALIZATION
# 1) Education vs Loan Status (1-> Loan Approved, 0-> Loan Not Approved)
sns.countplot(x='Education', hue='Loan_Status', data=loanData)
plt.title("Education vs Loan Status")
plt.show()

# 2) Marital Status vs Loan Status
sns.countplot(x='Married', hue='Loan_Status', data=loanData)
plt.title("Marital Status vs Loan Status")
plt.show()


# ALL PREPROCESSING AND VISUALISATION IS DONE

# Converting Categorical Data into Numerical Data

# Married --> YES: 1, NO: 0
# Gender --> Male 1, Female 0
# Self_Employed --> Yes: 1, No: 0
# Property_Area --> Urban: 2, Semiurban: 1, Rural: 0
# Education --> Graduate: 1, Not Graduate: 0

loanData.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Female': 0, 'Male': 1},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Not Graduate': 0, 'Graduate': 1}
}, inplace=True)
# print(loanData.head())

# Ensure target is int for classification
loanData['Loan_Status'] = loanData['Loan_Status'].astype(int)

a = loanData.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
b = loanData['Loan_Status']


# Splitting the data into training and testing data
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

scaler = StandardScaler()
a_train = scaler.fit_transform(a_train)  
a_test = scaler.transform(a_test)        

# Training the model
model= svm.SVC(kernel='linear')
model.fit(a_train, b_train)

# Accuracy Score(Training Data)
trainPrediction = model.predict(a_train)
accuracy = accuracy_score(trainPrediction, b_train)
print("Accuracy of Training Data: ", accuracy)

# Accuracy Score(Testing Data)
testPrediction = model.predict(a_test)
accuracy = accuracy_score(testPrediction, b_test)
print("Accuracy of Testing Data: ", accuracy)


# Making a predictive system

# Sample input (replace these values with actual input)
input_data = ['Male', 'No', 0, 'Graduate', 'No', 5000, 2000, 150, 360, 1, 'Urban']
input_array = np.array(input_data).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=a.columns)

encoder = LabelEncoder()

# Label Encoding (same as training)
for col in input_df.columns:
    if input_df[col].dtype == 'object':
        input_df[col] = encoder.fit_transform(input_df[col].astype(str))

# Fill missing values if any
for col in input_df.columns:
    input_df[col] = input_df[col].fillna(a[col].median())

# Scale input features
input_scaled = scaler.transform(input_df)


prediction = model.predict(input_scaled)
if prediction[0] == 1:
    print("Loan Status: Approved ")
else:
    print("Loan Status: Not Approved ")


