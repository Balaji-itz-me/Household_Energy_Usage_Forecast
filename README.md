# 🔋 Household Energy Usage Forecast

A complete machine learning pipeline to forecast household power consumption using historical smart meter data. This project focuses on feature-rich modeling, robust validation, and insightful interpretation using Gradient Boosting and Neural Networks.

---

## 📌 Project Objectives

- Predict household **Global Active Power Consumption** accurately
- Engineer meaningful features like **time of day**, **season**, and **sub-metering**
- Build and compare multiple models: **Gradient Boosting**, **MLP Regressor**
- Tune hyperparameters using **RandomizedSearchCV**
- Visualize model performance and **feature importance**
- Ensure **reproducibility** and clean model saving

---

## 🧠 Models Used

- Linear Regression (baseline)
- Random Forest Regressor
- **Gradient Boosting Regressor** (best model)
- MLP Regressor (Neural Network)

---

## 📊 Evaluation Metrics

| Metric | Description                          | Best Model Value |
|--------|--------------------------------------|------------------|
| **MAE** | Mean Absolute Error                 | 0.0999           |
| **RMSE** | Root Mean Squared Error            | 0.1543           |
| **R²**  | Coefficient of Determination        | 0.8588           |

All models were validated using **5-Fold Cross-Validation** with consistent performance.

---

## 📁 Dataset

The dataset contains:
- Smart meter readings for **active/reactive power**, **voltage**, and **sub-metering**
- Engineered features:
  - `hour`, `day_of_week`, `month`, `season`
  - `is_peak_hour`, `is_daytime`, `part_of_day_*`
  - Transformed target: `Global_active_power_log`

Outliers were filtered using the **3 × IQR rule**.

---

## 📈 Visualizations

### 📌 Feature Importance  
![Feature Importance](visuals/feature_importance.png)

### 📊 Actual vs Predicted Plot  
![Actual vs Predicted](visuals/actual_vs_predicted.png)

### 🌀 Residual Distribution  
![Residual Plot](visuals/residual_plot.png)

---

## 🧪 Key Techniques

- Feature transformations: `log`, `sqrt`, `boxcox`
- Cross-validation with `cross_val_score`
- Hyperparameter tuning using `RandomizedSearchCV`
- Model persistence with `joblib`
- Visual diagnostics to assess model fit

---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/Balaji-itz-me/Household_Energy_Usage_Forecast.git
cd Household_Energy_Usage_Forecast
# Household_Energy_Usage_Forecast
```

## Create a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

## Install dependencies

pip install -r requirements.txt

## Usage
Notebooks are available under /notebooks for:

Data Cleaning & EDA

Feature Engineering

Model Training & Evaluation

Final Evaluation & Visualizations

To load the trained model:

import joblib
model = joblib.load("models/best_gradient_boosting_model.pkl")

##  Project Structure

``` Household_Energy_Usage_Forecast/
├── data/                      # Raw & processed data
├── notebooks/                 # Jupyter notebooks
├── models/                    # Saved models
├── src/                       # Scripts and helper functions
├── visuals/                   # PNG plots and visualizations
├── requirements.txt           # Dependencies
└── README.md
```

## License

This project is licensed under the MIT License.

## 🤝 Acknowledgments
Open-source smart meter dataset

scikit-learn, pandas, matplotlib, seaborn

Community support for machine learning best practices

## 🙋‍♂️ Author
BALAJI K
📫 GitHub Profile
🛠️ Data Science Enthusiast | Python Developer | Model Tuner
