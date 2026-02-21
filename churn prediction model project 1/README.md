Customer Churn Prediction using Machine Learning

Project Overview

This project predicts whether a telecom customer will churn using machine learning techniques.

The dataset contains 7,043 customer records with demographic, subscription, and billing information.

Technologies Used

• Python
• Pandas
• NumPy
• Scikit-learn
• Imbalanced-learn (SMOTE)
• XGBoost


customer-churn-ml/
│
├── data / Telco-Customer-Churn.csv
│
├── notebooks / churn_eda.ipynb
│
├── src / train.py
│
├── models / customer_churn_model.pkl
│
├── requirements.txt
└── README.md


Workflow

1.	Data Cleaning
2.	Feature Encoding
3.	Train-Test Split
4.	SMOTE for class imbalance
5.	Model Training (Random Forest)
6.	Model Evaluation
7.	Model Saving