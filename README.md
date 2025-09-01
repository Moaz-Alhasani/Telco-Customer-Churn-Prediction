# 📊 Telco Customer Churn Prediction

A project to predict **Customer Churn** using Machine Learning algorithms such as Decision Tree, Random Forest, and XGBoost.  
The project applies **data preprocessing**, **balancing data with SMOTE**, and **hyperparameter tuning** using RandomizedSearchCV.

---

## 🚀 Features
- Handle missing values and categorical encoding.
- Exploratory Data Analysis (EDA) with visualizations.
- Train multiple ML models and compare performance.
- Hyperparameter optimization with RandomizedSearchCV.
- Save the best model and encoders for later use.
- Make predictions on new input data.

---

## 📂 Project Structure
```
📦 Telco-Churn-Prediction
┣ 📜 WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
┣ 📜 customer_churn_model.pkl             # Saved best model
┣ 📜 encoders.pkl                         # LabelEncoders for categorical columns
┣ 📜 churn_prediction.py                  # Main Python script
┗ 📜 README.md                            # Documentation file
```

---

## 🛠️ Requirements
- Python 3.8+
- Libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

---

## 📊 Training and Evaluation
- **Models trained**: Decision Tree, Random Forest, XGBoost  
- **Evaluation metrics**: Accuracy, Confusion Matrix, Classification Report  
- **Data imbalance** handled using **SMOTE**  

---

## 🔮 Usage

1. Train the model with the dataset.  
2. The best model is automatically saved as `customer_churn_model.pkl`.  
3. Encoders are saved in `encoders.pkl`.  
4. To make a prediction on new data:

```python
import pickle
import pandas as pd

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
new_data = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}])

prediction = model.predict(new_data)
print("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
```

📌 **Example Prediction Output**
```bash
Prediction: Churn
Prediction Probability: [[0.32, 0.68]]
```


