from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
from preprocessing.transformer import load_and_preprocess_training_data
from config import MODEL_PATH

# 1. Load & preprocess data
X, y = load_and_preprocess_training_data()

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Grid search for Decision Tree
dt = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4],
    'max_features': [None, 'sqrt', 'log2']
}
grid = GridSearchCV(
    dt, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

best_dt = grid.best_estimator_
print("Best params:", grid.best_params_)

# 4. Evaluate
y_pred = best_dt.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):.2f}")
print(f"Test MAE : {mean_absolute_error(y_test,y_pred):.2f}")
print(f"Test R²  : {r2_score(y_test,y_pred):.4f}")

# 5. Save model
joblib.dump(best_dt, MODEL_PATH)
print("Model saved to", MODEL_PATH)