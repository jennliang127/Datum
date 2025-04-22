# Machine Learning Regression Project

## Project Overview
This project explores different regression models to predict median house value.  
The objective is to build, evaluate, and compare multiple models to identify the best-performing algorithm.

## Dataset
- Source: ""https://raw.githubusercontent.com/ageron/handson-ml2/datasets/housing/housing.csv"

- Features: 
|   |#   Column          |Non-Null|  Count | Dtype | 
|---|--------------------|--------|--------|--------|  
| 0   longitude          |20640   |non-null| float64|
| 1   latitude           |20640   |non-null| float64|
| 2   housing_median_age |20640   |non-null| float64|
| 3   total_rooms        |20640   |non-null| float64|
| 4   total_bedrooms     |20433   |non-null| float64|
| 5   population         |20640   |non-null| float64|
| 6   households         |20640   |non-null| float64|
| 7   median_income      |20640   |non-null| float64|
| 8   median_house_value |20640   |non-null| float64|
| 9   ocean_proximity    |20640   |non-null| object |
| dtypes: float64(9), object(1)
| memory usage: 1.6+ MB


- Target: Median_house_value 

## Models Trained
- Linear Regression
- Ridge Regression
- SGD Regressor
- Lasso Regression
- Elastic Net Regressor
- Bayesian Ridge Regressor
- Kernel Ridge Regressor
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor


## Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R² Score)

## Model Comparison
| Model                    | MAE  | MSE  | RMSE | R² Score |
|---------------------------|------|------|------|----------|
| Linear Regression         | ...  | ...  | ...  | ...      |
| Ridge Regression          | ...  | ...  | ...  | ...      |
| SGD Regressor             | ...  | ...  | ...  | ...      |
| Lasso Regression          | ...  | ...  | ...  | ...      |
| Elastic Net               | ...  | ...  | ...  | ...      |
| Bayesian Ridge Regressor  | ...  | ...  | ...  | ...      |
| Kernel Ridge Regressor    | ...  | ...  | ...  | ...      |
| Random Forest Regressor   | ...  | ...  | ...  | ...      |
| Decision Tree Regressor   | ...  | ...  | ...  | ...      |
| Gradient Boosting Regressor| ... | ...  | ...  | ...      |
| XGBoost Regressor         | ...  | ...  | ...  | ...      |

## Conclusion
Based on the evaluation metrics, [Model Name] performed the best in predicting [target variable].  
Future improvements could include hyperparameter tuning, feature engineering, or model stacking.

