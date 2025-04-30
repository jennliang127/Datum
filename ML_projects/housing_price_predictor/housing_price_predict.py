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
            'search Method': name
            'MSE': mse,
            'RMSE':rmse,
            'R2': r2,
            'Cross value score mean':cvs_rmse_scores.mean(),
            'Cross value score std':cvs_rmse_score.std()
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


# Fine Tuning: 

param_grids = [{'alphas': np.logspace(-6,1,100),
               'max_iter':(50, 1000),
               'max_depth':[3,5,7],
               'tol':(1e-6, 1e-2,'long-uniform')
               }]
    

def run_fine_tuning(model, param_grids, x_train, y_train):
    results = {}
    
    searches = {
        "GridSearchCV": GridSearchCV(model, param_grids, cv=5, n_jobs=-1),
        "RandomizedSearchCV": RandomizedSearchCV(model, param_grids['random'], n_iter=20, cv=5, n_jobs=-1),
        "BayesSearchCV": BayesSearchCV(model, param_grids['bayes'], n_iter=20, cv=5, n_jobs=-1),
        "gp_minimize": 
        "dummy_minimize":
        "forest_minimize":
    }
    
    for name, searcher in searches.items():
        print(f"Running {name}...")
        start = time()
        searcher.fit(x_train, y_train)
        elapsed = time() - start
        results[name] = {
            "best_score": searcher.best_score_,
            "best_params": searcher.best_params_,
            "time_taken": elapsed
        }
        
    return results