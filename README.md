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
* Final RMSE, MAE, R² for each model

| Model                              | RMSE (₹) | MAE (₹) | R²     | CV R² |
| ---------------------------------- | -------- | ------- | ------ | ----- |
| **Linear Regression (original y)** | 3,494.4  | 2,586.2 | 0.7136 | 0.617 |
| **Decision Tree (default)**        | 3,013.4  | 1,333.3 | 0.7870 | 0.500 |
| **Decision Tree (tuned)**          | 2,755.1  | 1,757.3 | 0.8220 | 0.668 |
| **Random Forest (untuned)**        | 2,421.4  | 1,439.5 | 0.8625 | 0.721 |
| **Gradient Boosting (untuned)**    | 2,487.1  | 1,675.1 | 0.8549 | 0.735 |

**Random Forest** remains the best off-the-shelf choice. Its ensemble nature better balances bias/variance.

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

## Insights

1. **Data Integrity & Size**  
   - Our dataset of 986 records and 13 features is complete—**no missing values**—and appropriately sized for both interpretable linear models and more flexible tree-based methods.  

2. **Outlier Patterns**  
   - **Binary flags** (e.g. chronic disease, transplants, allergies, family cancer) each have IQR=0, so IQR‐based outlier removal would eliminate every “1.”  We therefore isolate outlier handling to continuous features.  
   - Among continuous features:  
     - **Weight** has ~1.6 % of values above 117 kg.  
     - **BMI** has ~2.2 % beyond [12.3, 41.8].  
     - **Surgery count** sees ~1.6 % of “3” cases.  
     - **PremiumPrice** has ~0.6 % above ₹38 500.  
   - Visualizing before/after shows that **capping** or **transforming** these tails (rather than dropping records) preserves important high-risk cases while reducing undue influence on variance.

3. **Univariate & Bivariate Distributions**  
   - **Age** is uniformly distributed across 18–66, with no extreme outliers.  
   - Only ~18 % report chronic diseases, ~6 % transplants, ~47 % blood-pressure issues.  These imbalances signal important risk flags without overwhelming the model.  
   - **PremiumPrice** is right-skewed (₹15 000–₹30 000 bulk, tail to ₹40 000). A log-transform or robust estimator will tame that skew.  
   - **Premium vs. Chronic Disease**: chronic-disease holders pay ~₹4 000–₹5 000 more (IQR shift).  
   - **Premium vs. Diabetes**: diabetics show moderate uplift (~₹2 000) and fatter right tail.  
   - **Premium vs. Age**: clear linear trend—older age groups regularly incur higher costs.  
   - **Premium vs. BMI**: a weak but noticeable upward drift at extreme BMI values.

4. **Correlation Structure**  
   - **Age (r≈0.70)** is the strongest linear predictor of premium.  
   - Among binary flags, **transplants (r≈0.29)**, **surgery count (r≈0.26)** and **chronic diseases (r≈0.21)** are the next most correlated.  
   - **Diabetes** and **family cancer history** each have only weak direct correlation (r≈0.08), hinting at non-linear or interaction effects.

5. **Formal Hypothesis Tests**  
   - **T-tests** confirm that premiums are significantly higher for applicants with diabetes, blood-pressure problems, transplants, chronic diseases, and family cancer history (all p<0.05)—but **not** for allergies (p≈0.71).  
   - **ANOVA** shows surgery count groups differ in mean premium (F=26.1, p<10⁻¹⁶), validating it as an ordinal risk factor.  
   - **Chi-square tests** reveal that most binary flags are statistically independent (e.g. chronic diseases vs. family cancer, p≈0.89), so they contribute distinct information.

6. **Baseline OLS Regression**  
   - A model with **Age**, **BMI**, **Diabetes**, **AnyChronicDiseases** and **NumberOfMajorSurgeries** achieves R²≈0.54.  
   - **Age** adds ~₹323 per year; **BMI** adds ~₹151 per point; **chronic diseases** add ~₹2 853.  
   - The **negative diabetes coefficient** (–₹663) and non-significant surgery term suggest collinearity or that these risks overlap with age/BMI/chronic conditions.

7. **Model Comparisons**  
   - **Random Forest (untuned)** outperforms all: **test R² ≈ 0.86**, **RMSE ≈ ₹2 421**, with stable CV R² ≈ 0.72.  
   - **Gradient Boosting** comes close (R² ≈ 0.85, RMSE ≈ ₹2 487), while **tuned Decision Trees** improve over default but still lag.  
   - **Linear Regression** remains the most interpretable baseline (R² ≈ 0.71, RMSE ≈ ₹3 494) but has higher error.

---

## Actionable Recommendations

1. **Outlier Treatment**  
   - **Winsorize** or **log-transform** continuous features (Weight, BMI, PremiumPrice) at their IQR bounds to reduce tail influence without losing rare high-risk cases.  

2. **Feature Engineering**  
   - Introduce **polynomial** (e.g. Age²) and **interaction** terms (e.g. BMI × chronic disease) to capture non-linear and compounding effects.  
   - Consider grouping “3+ surgeries” as its own high-risk bucket rather than treating it as an extreme outlier.

3. **Modeling Strategy**  
   - Use **Random Forest** or **Gradient Boosting** as primary predictors for final deployment, given their superior accuracy and robustness to skew/outliers.  
   - Retain a **linear model** as a transparent fallback for regulatory or interpretability needs.

4. **Target Transformation**  
   - Apply **log(PremiumPrice)** in regression models to stabilize variance and satisfy homoscedasticity/normality assumptions.

5. **Validation & Robustness**  
   - Employ **k-fold cross-validation** on all candidate models (including regularized and robust regressors) to guard against overfitting.  
   - Use **permutation importance** or **SHAP values** on tree-based models to explain predictions at both the global and individual levels.

---

## Future Work

- **Expand Feature Set**: Incorporate additional health indicators (e.g. smoking status, cholesterol levels) or socio-demographic variables to capture more variance.  
- **Time-Series Extension**: If longitudinal data become available, build dynamic models that adjust premiums over policy renewal cycles.  
- **Automated Retraining**: Implement a pipeline (e.g. with Airflow or GitHub Actions) that retrains and redeploys models as new data arrive.  
- **Fairness & Bias Auditing**: Evaluate model performance across demographic subgroups to ensure equitable pricing.  
- **Production Deployment**: Complete the **Flask API** and **Streamlit app** end-to-end deployment, including monitoring and logging for real-time predictions.  
- **Advanced Algorithms**: Explore **XGBoost**, **LightGBM**, or **deep learning** for any remaining gains, and compare against our current random-forest benchmark.

By following these recommendations and future directions, we can deliver a predictive system that is accurate, robust, interpretable, and ready for operational use in insurance underwriting.  