# Machine Learning Regression Project

## Project Overview
This project explores different regression models to predict median house value.  
The objective is to build, evaluate, and compare multiple models to identify the best-performing prediction algorithm.

## Dataset
- Source: ""https://raw.githubusercontent.com/ageron/handson-ml2/datasets/housing/housing.csv"


# Features: 
Data structure:

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

- Target prediction: Median_house_value 

## Models Trained
Following 9 regression models are choosen to conduct the prediction training. The goal is to understand which regression model returns the best prediction with minimized root-mean-square error, maximized R-square test, and smallest dispersion in cross value score.

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

## Note: 
Based on the evaluation metrics, before fine tune the regression models, Lasso Regression, Linear Regression and Bayesian Regression return the best performance in predicting the median housing values.

However, I wish to check if a better prediction performance can be achieved by hyperparameter tuning. The top three performing models are been selected to the fine-tune process. Following are a list of hyper tunning methods to explore.

## Fine Tunning Method

- RandomizedSearchCV (fast)
    - Best for medium or big space, and fast trail 
    - Best for **Tree-based models**, large models
    - It tires random points, however deosn't guarantee best optimalization. Can be a good trade off depends on the tolerance

    
- BayesSearchCV (medium fast)
    - Best for medium or large space search
    - Best for **XGBoost**, **LightGBM**, **SVR**, **ensemble** methods
    - It learns where good areas are, and conduct a smarter search than random

- GridSearchCV (slow)
    - Best for small search space
    - Best for **Linear** models, **Ridge**, **Lasso**, small **trees**
    - Relatively slow, it tests all combinations, and it guaranteed to   find the best inside the grid, but at an expense


- Gaussian Process (gp_minimize)(medium fast)
    - Best for small space search
    - best for **continuous hyperparameter** spaces
    - It is more computationally expensive, it works well for smooth data and expensive models

- Random Search baseline (dummy_minimize) (fast)
    - Best for testing and benchmarking 
    - Best for **Random Forests**, **LightGBM**
    - This search method is similar to random guessing, frequently use as comparison baseline

- forest_minimize (ok fast)
    - Best for noisy functions with large space 
    - Best use for high-dimensional spaces search

Following is the performance result after hyperparameter tuning with each method. 

## Fine Tuned performance
| Tuning method             | Model                    | Best Parameters| RMSE      | R² Score | CV score (mean)| CV score (STD)|
|---------------------------|--------------------------|----------------|-----------|----------|----------------|---------------|
| RandomizedSearchCV        | Lasso Regression         |                |           |          |                |               |
| BayesSearchCV             | Bayesian Ridge Regressor |                |           |          |                |               |
| GridSearchCV              | Linear Regression        |                |           |          |                |               |
| gp_minimize (Gaussian Process) | Bayesian Ridge Regressor |           |           |          |                |               |
| dummy_minimize (Random search baseline)| Lasso Regression  |          |           |          |                |               |
| forest_minimize           | Linear Regression        |                |           |          |                |               |


