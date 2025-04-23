# Machine Learning Regression Project

## Project Overview
This project explores different regression models to predict median house value.  
The objective is to build, evaluate, and compare multiple models to identify the best-performing algorithm.

## Dataset
- Source: ""https://raw.githubusercontent.com/ageron/handson-ml2/datasets/housing/housing.csv"

# Features: 
|  Column          |Non-Null|  Count | Dtype | 
|-----------------------|--------|--------|--------|  
| longitude          |20640   |non-null| float64|
| latitude           |20640   |non-null| float64|
| housing_median_age |20640   |non-null| float64|
| total_rooms        |20640   |non-null| float64|
| total_bedrooms     |20433   |non-null| float64|
| population         |20640   |non-null| float64|
| households         |20640   |non-null| float64|
| median_income      |20640   |non-null| float64|
| median_house_value |20640   |non-null| float64|
| ocean_proximity    |20640   |non-null| object |
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


## Evaluation Metrics
- Cross Value Score (CVS)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R² Score)

## Model Comparison (Before Tuned)
| Model                     | CVS (Mean)  | CVS (STD)| RMSE | R² Score |
|---------------------------|------|------|------------|----------|
| Linear Regression         | 2569  |68587.45 |68855.75   | 0.649  |
| Ridge Regression          | 68586.79  |2552.62 | 68856.597   |0.649 |
| SGD Regressor             | 14981042.39 |26331419.58 | 68857.45  | 0.649  |
| Lasso Regression          | 68587.46  |2566.51| 68855.03  | 0.649   |
| Elastic Net Regressor     | 78704.02  |1465.50| 79645.374  | 0.530  |
| Bayesian Ridge Regressor  | 68595.19  |2539.80 | 68856.27  | 0.649  |
| Kernel Ridge Regressor    | 68606.01  |2533.65 | 68891.80  | 0.648  |
| Random Forest Regressor   | 49328.66  |1800.69| 70487.99  | 0.632   |
| Decision Tree Regressor   | 71027.46  |1992.45| 96314.41  | 0.313   |

## Conclusion
Based on the evaluation metrics, before fine tune the regression model, performed the best in predicting median housing price.  
Future improvements could include hyperparameter tuning, feature engineering, or model stacking.

