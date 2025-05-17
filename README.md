# Health Insurance Cost Prediction

## Problem Statement
Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. Traditional methods of cost prediction often rely on broad actuarial tables and historical averages, which may not account for the nuanced differences among individuals. By leveraging machine learning techniques, insurers can predict more accurately the insurance costs tailored to individual profiles, leading to more competitive pricing and better risk management.

## Target Metric
- **Root Mean Squared Error (RMSE)** on the predicted annual premium price.  
- **R² (Coefficient of Determination)** to assess the proportion of variance explained.

## Dataset Description
The dataset comprises the following 11 attributes:
1. **Age** (`int`, 18–66)  
2. **Diabetes** (`0`/`1`): presence of diabetes  
3. **BloodPressureProblems** (`0`/`1`): history of blood pressure issues  
4. **AnyTransplants** (`0`/`1`): indicates prior transplants  
5. **AnyChronicDiseases** (`0`/`1`): presence of chronic diseases  
6. **Height** (`cm`, 145–188)  
7. **Weight** (`kg`, 51–132)  
8. **KnownAllergies** (`0`/`1`)  
9. **HistoryOfCancerInFamily** (`0`/`1`)  
10. **NumberOfMajorSurgeries** (`int`, 0–3)  
11. **PremiumPrice** (`numeric`, 15,000–40,000): annual insurance premium (target)

---

## Steps Taken

### 1. Exploratory Data Analysis & Hypothesis Testing
- **Distribution Analysis**  
- **Correlation Matrix & Heatmap**  
- **Outlier Detection (IQR / Z-score)**  
- **Hypothesis Tests**  
  - T-tests / ANOVA (e.g. smokers vs. non-smokers; surgery groups)  
  - Chi-square (chronic disease vs. family cancer history)  
  - Regression-based tests  

### 2. Feature Engineering & Preprocessing
- Handling missing values (if any)  
- Created **BMI** from height & weight  
- Scaling numerical features & encoding binaries  

### 3. Machine Learning Modeling & Validation
- **Baseline**: Linear Regression  
- **Tree-based**: Decision Tree, Random Forest, Gradient Boosting (untuned & tuned)  
- **Neural Network**: MLPRegressor (tuned)  
- **Cross-Validation**: 5-fold CV  
- **Metrics**: RMSE, MAE, R²  
- **Confidence/Prediction Intervals** from OLS  

### 4. Interpretability & Explainability
- **Permutation Feature Importance**  
- **SHAP Values**  
- **Decision-tree rule extraction** for transparent underwriting  

### 5. Deployment
- **Flask API**  
  - `/api/predict` endpoint (JSON in → premium out)  
- **Streamlit App**  
  - Interactive web calculator  
- **Hosting**  
  - Streamlit Community Cloud  

---

## Final Model Scores
*(To be populated with your final RMSE, MAE, R² for each model)*

| Model                       | CV R²  | Test R² | Test RMSE | Test MAE |
|-----------------------------|-------:|--------:|----------:|---------:|
| Linear Regression           |        |         |           |          |
| Decision Tree (tuned)       |        |         |           |          |
| Random Forest (untuned)     |        |         |           |          |
| Random Forest (tuned)       |        |         |           |          |
| Gradient Boosting (tuned)   |        |         |           |          |
| Neural Network (MLP, tuned) |        |         |           |          |

---

## Deployment Steps
1. **Save** `scaler.joblib` & `tuned_model.joblib` in `models/`  
2. **Flask**  
   - Install dependencies (`requirements.txt`)  
   - Run `python app.py` → API live at `http://localhost:5000/api/predict`  
3. **Streamlit**  
   - Run `streamlit run streamlit_app.py` → local UI  
   - Push to GitHub & deploy on Streamlit Community Cloud  

---

## Requirements
- Python ≥ 3.8 
- pandas, numpy, scikit-learn, statsmodels, scipy  
- Flask, Streamlit, joblib  

---

## Insights & Recommendations
*(To be populated after analysis)*

- **Key Risk Drivers**: e.g. Age, Chronic Diseases, BMI  
- **Business Actions**: e.g. tiered pricing, wellness incentives  

---

## Future Work
- More advanced modeling (XGBoost, deep learning)  
- Automated retraining pipeline  
- Integration with live claims data  
