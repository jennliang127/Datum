# Copyright (c) 2025 Jenn Liang

from imports import * 
from DataPrep import DataPrep
from time import time
from CombinedAttributesAdder import CombinedAttributesAdder

# Create class instance: 
dp = DataPrep()

"""
Extrapolate Data Analysis (EDA):
"""
# read csv data as DataFrame
file_name = 'housing.csv'
read_file = os.path.join(os.getcwd(),f'data/{file_name}')
housing_raw = pd.read_csv(read_file)


# Check data structure and descriptive statistics: 

housing_raw.info()
housing_raw.keys()
housing_raw.head()
housing_raw.describe()

housing_raw['ocean_proximity'].value_counts() 

# Check distribution data
housing_raw.hist(bins=50) # Create histogram of raw housing data
                          ## Note: median_income and median_house_value have 
                          #        a skewed distribution. total_rooms, tota_bedrooms,
                          #        population, households have similar distributions 
plt.tight_layout()
plt.show()

"""
Data Preparation for Machine Learning model 
"""
# Plot correlations between each numerical attribute against 'median_house_value'
num_attrib = dp.plot_num_feature_corr(housing_raw,'median_house_value')

# Split data into test set and training set: 
train_data, test_data = dp.split_train_test(housing_raw,0.2)

# Data Transformation
# Replace missing value with most frequent values from the column
imputer = SimpleImputer(strategy="most_frequent") 

# Consider add bedrooms per room as training parameter (True, False)
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)

# Add bedrooms per room value as part of traning parameters
housing_extra_attribs=attr_adder.transform(train_data.values)


# Separate Numerical Attributes and Categorical Attribute
num_dtypes = ['int16','int32','int64',
              'float16','float32','float64']
housing_train = housing_raw.drop("median_house_value",axis=1)
num_attribs = [i for i in housing_train.keys() if housing_train[i].dtype in num_dtypes]
cat_attribs = ["ocean_proximity"]

# Define data transform method to numerical attributes
num_pipeline = Pipeline([('imputer', imputer),
                         ('attribs_adder', attr_adder),
                         ('std_scalar', StandardScaler()),
                         ])

# Define transformation pipline to Numertical Attributes and Catagorical Attribute
full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',OneHotEncoder(),cat_attribs),
    ])

# Prepare training data drop and copy intended predict filed
housing_train_data = train_data.drop("median_house_value",axis=1)
cp_training_housing_reported = train_data["median_house_value"].copy()

housing_test_data = test_data.drop("median_house_value",axis=1)
cp_test_housing_reported = test_data["median_house_value"].copy()

# Preparing training data with full transformation pipline
housing_train_prepared = full_pipeline.fit_transform(housing_train_data)
housing_test_prepared = full_pipeline.fit_transform(housing_test_data)

# Regressions: 
lin_reg = LinearRegression()
ridge_reg = Ridge()
sgd_reg = SGDRegressor()
lass_reg = Lasso()
elas_reg = ElasticNet()
bayes_ridge = BayesianRidge()
kern_ridge = KernelRidge()
forest_reg = RandomForestRegressor()
dectree_reg = DecisionTreeRegressor()

svr_reg = SVR()

def evaluate_regression(regressions, x_train, y_train, x_test, y_test):
    reg_results = {}
    for reg, reg_model in regressions.items():
        reg_model.fit(x_train, y_train)
        reg_predic = reg_model.predict(x_test)
        
        # Evaluate performance
        mse = mean_squared_error(y_test, reg_predic) # mean square error
        rmse = np.sqrt(mse)                          # Square root MSE
        r2 = r2_score(y_test, reg_predic)            # R-square test
        cvs = cross_val_score(reg_model, x_test, y_test, 
                              scoring="neg_mean_squared_error", cv=10)    # Cross value score
        cvs_rmse_scores = np.sqrt(-cvs)
        reg_results[reg] = {
            'search Method':reg,
            'MSE': mse,
            'RMSE':rmse,
            'R2': r2,
            'Cross value score mean':cvs_rmse_scores.mean(),
            'Cross value score std':cvs_rmse_scores.std()
        }
    return reg_results

regressions = {'Linear Regression': lin_reg,
               'Ridge Regression':ridge_reg,
               'SGD Regression':sgd_reg,
               'Lasso Regression': lass_reg,
               'Elastic Net Regression': elas_reg,
               'Bayesian Ridge Regression':bayes_ridge,
               'Kernel Ridge Regression':kern_ridge,
               'Random Forest Regression':forest_reg,
               'Decision Tree Regression':dectree_reg}

results = evaluate_regression(regressions, 
                              housing_train_prepared, 
                              cp_training_housing_reported,
                              housing_test_prepared, 
                              cp_test_housing_reported)


# Test Hyperparameter Tuning: 

"""
param_grids configs the initial condition to search for optimal solution 

"""

param_grids = {'random': {
                    'n_estimators': [50, 100, 200],
                    'max_depth':[3,5,7, None],
                    'min_samples_split':[2, 5, 10],
                    'bootstrap':[True, False]
               },
                'bayes': {
                    'alpha': Real(1e-3, 1e1, prior='log-uniform'),
                    'fit_intercept': Categorical([True,False]),
                },

                'lin_reg_gp_min':[Categorical([True, False],name = 'fit_intercept'),
                                  Categorical([True, False], name = 'positive'),
                                  Categorical([True, False], name = 'copy_X')
                                 ],
                #            Integer(3,10, name = 'max_depth'),
                #            Integer(2,10, name = 'min_smaples_split'),

                # 'dummy_min': [ Integer(50,200, name = 'n_estimators'),
                #                Integer(3,10, name='max_depth'),
                #                Integer(2,10, name='min_samples_split')
                # ],
                
                # 'forest_min': [ Integer(50,200, name = 'n_estimators'),
                #                 Integer(3,10, name = 'max_depth'),
                #                 Integer(2,10, name = 'min_smaples_split')
                # ],
                'skopt': {'alpha':[0.001, 1, 10],
                        'fit_intercept': [True]
                },

                'lin_reg_grid':{'copy_X':[True, False],
                                'fit_intercept': [True, False],
                                'n_jobs': [1,2,3],
                                'positive': [True,False]},

                'lin_reg_rand':{'fit_intercept': [True, False],
                                'positive': [True, False]},

                'lin_reg_bayes':{'fit_intercept': Categorical([True, False]),
                                 'positive': Categorical([True, False]),
                                 'copy_X': Categorical([True, False])},
                
                'lass_grid_rand':{'alpha':[1e-3,1e-2,1e-1,1e0,1e1],
                                  'fit_intercept':[True, False],
                                  'max_iter': [1000, 5000, 100000],
                                  'tol': [1e-4, 1e-3],
                                  'selection': ['cyclic', 'random']},

                'lass_bayes':{'alpha':Real(1e-4, 10e1, prior='log-uniform'),
                              'fit_intercept':Categorical([True, False]),
                              'max_iter': Integer(1000, 100000),
                              'tol': Real(1e-5, 1e-2,prior='log-uniform'),
                              'selection': Categorical(['cyclic', 'random'])},

                'bayes_grid_rand':{'alpha_1':[1e-6, 1e-5, 1e-4],
                                   'alpha_2':[1e-6, 1e-5, 1e-4],
                                   'lambda_1':[1e-6, 1e-5, 1e-4],
                                   'lambda_2':[1e-6, 1e-5, 1e-4],
                                   'fit_intercept':[True, False],
                                   'tol': [1e-4, 1e-3]
                                   },

                'bayes_bayes':{'alpha_1': Real(1e-6, 1e-3, prior='log-uniform'),
                               'alpha_2': Real(1e-6, 1e-3, prior='log-uniform'),
                               'lambda_1': Real(1e-6, 1e-3, prior='log-uniform'),
                               'lambda_2': Real(1e-6, 1e-3, prior='log-uniform'),
                               'fit_intercept':Categorical([True, False]),
                               'tol': Real(1e-5, 1e-2, prior='log-uniform')

                }
}


def run_linear_regression_tuning(model, param_grids, x_train, y_train, 
                    scoring='neg_mean_squared_error'):

    results = {}
    
    searches = {
        "GridSearchCV": GridSearchCV(model, param_grids['lin_reg_grid'], cv=5, n_jobs=-1),
        "RandomizedSearchCV": RandomizedSearchCV(model, param_grids['lin_reg_rand'], n_iter=20, cv=5, n_jobs=-1),
        "BayesSearchCV": BayesSearchCV(model, param_grids['lin_reg_bayes'], n_iter=20, cv=5, n_jobs=-1)    
    }
    
    for name, searcher in searches.items():
        print(f"Searching with {name} ...")
        start = time()
        searcher.fit(x_train, y_train)
        elapsed = time() - start
        results[name] = {
            "best_score": searcher.best_score_,
            "best_params": searcher.best_params_,
            "time_taken": elapsed
        }
    
    param_names = [dim.name for dim in param_grids['lin_reg_gp_min']] 

    # Wrap object for skopt minimize-based searchers
    def object_wrapper(params):
        param_dict = dict(zip(param_names, params))
        model.set_params(**param_dict)
        scores = cross_val_score(model, x_train, y_train, cv=5, scoring = scoring, n_jobs = -1)
        return -np.mean(scores)

    skopt_dict = {
        "gp_minimize": gp_minimize,
        "dummy_minimize": dummy_minimize,
        "forest_minimize": forest_minimize
        }

    for method_name, methods in skopt_dict.items():
        print(f"Searching with {method_name} ...")
        start = time()
        result = methods(object_wrapper, param_grids['lin_reg_gp_min'],
                        n_calls = 200, random_state = 0)
        elaspsed = time() -start
        best_param = dict(zip(param_names, result.x))
        results[method_name] = {
            "best_score":result.fun,
            "best_params":dict(zip(param_names, result.x)),
            "time elapsed": elapsed
        }
        
        
    return results


def run_lasso_regression_tuning(model, param_grids, x_train, y_train, 
                    scoring='neg_mean_squared_error'):

    results = {}
    searches = {
        "GridSearchCV": GridSearchCV(model, param_grids['lass_grid_rand'], cv=5, n_jobs=-1),
        "RandomizedSearchCV": RandomizedSearchCV(model, param_grids['lass_grid_rand'], n_iter=20, cv=5, n_jobs=-1),
        "BayesSearchCV": BayesSearchCV(model, param_grids['lass_bayes'], n_iter=20, cv=5, n_jobs=-1)    
    }
    
    for name, searcher in searches.items():
        print(f"Searching with {name} ...")
        start = time()
        searcher.fit(x_train, y_train)
        elapsed = time() - start
        results[name] = {
            "best_score": searcher.best_score_,
            "best_params": searcher.best_params_,
            "time_taken": elapsed
        }


    param_names = [dim.name for dim in param_grids['lin_reg_gp_min']] 

    # Wrap object for skopt minimize-based searchers
    def object_wrapper(params):
        param_dict = dict(zip(param_names, params))
        model.set_params(**param_dict)
        scores = cross_val_score(model, x_train, y_train, cv=5, scoring = scoring, n_jobs = -1)
        return -np.mean(scores)

    skopt_dict = {
        "gp_minimize": gp_minimize,
        "dummy_minimize": dummy_minimize,
        "forest_minimize": forest_minimize
        }

    for method_name, methods in skopt_dict.items():
        print(f"Searching with {method_name} ...")
        start = time()
        result = methods(object_wrapper, param_grids['lin_reg_gp_min'],
                        n_calls = 200, random_state = 0)
        elaspsed = time() -start
        best_param = dict(zip(param_names, result.x))
        results[method_name] = {
            "best_score":result.fun,
            "best_params":dict(zip(param_names, result.x)),
            "time elapsed": elapsed
        }
    
    return results


def run_bayes_ridge_tuning(model, param_grids, x_train, y_train, 
                    scoring='neg_mean_squared_error'):

    results = {}
    searches = {
        "GridSearchCV": GridSearchCV(model, param_grids['bayes_grid_rand'], cv=5, n_jobs=-1),
        "RandomizedSearchCV": RandomizedSearchCV(model, param_grids['bayes_grid_rand'], n_iter=20, cv=5, n_jobs=-1),
        "BayesSearchCV": BayesSearchCV(model, param_grids['bayes_bayes'], n_iter=20, cv=5, n_jobs=-1)    
    }
    
    for name, searcher in searches.items():
        print(f"Searching with {name} ...")
        start = time()
        searcher.fit(x_train, y_train)
        elapsed = time() - start
        results[name] = {
            "best_score": -searcher.best_score_,
            "best_params": searcher.best_params_,
            "time_taken": elapsed
        }
    
    return results



linear_search_results = run_linear_regression_tuning(lin_reg, param_grids,
                                        housing_train_prepared, 
                                        cp_training_housing_reported)

lasso_search_results = run_lasso_regression_tuning(lass_reg, param_grids,
                                        housing_train_prepared, 
                                        cp_training_housing_reported)


bayesRidge_search_results = run_bayes_ridge_tuning(bayes_ridge, param_grids,
                                            housing_train_prepared, 
                                            cp_training_housing_reported)


# Using the fine tuned modle to make prediction 