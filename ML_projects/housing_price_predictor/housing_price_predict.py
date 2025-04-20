from imports import * 
from DataPrep import DataPrep
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

# Drop intend prediction field 
cp_housing = train_data.drop("median_house_value", axis=1)

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

# Preparing training data with full transformation pipline
housing_train_prepared = full_pipeline.fit_transform(housing_train_data)


# Selected Regressions: 
lin_reg = LinearRegression()
sgd_reg = SGDRegressor()
lass_reg = Lasso()
elas_reg = ElasticNet()
bayes_ridge = BayesianRidge()
kern_ridge = KernelRidge()
forest_reg = RandomForestRegressor()
dectree_reg = DecisionTreeRegressor()

svr_reg = SVR()

# Train data set with each regressions:
lin_reg.fit(housing_train_prepared, cp_training_housing_reported)
sgd_reg.fit(housing_train_prepared, cp_training_housing_reported)
lass_reg.fit(housing_train_prepared, cp_training_housing_reported)
elas_reg.fit(housing_train_prepared, cp_training_housing_reported)
bayes_ridge.fit(housing_train_prepared, cp_training_housing_reported)
kern_ridge.fit(housing_train_prepared, cp_training_housing_reported)

forest_reg.fit(housing_train_prepared, cp_training_housing_reported)
dectree_reg.fit(housing_train_prepared, cp_training_housing_reported)
svr_reg.fit(housing_train_prepared, cp_training_housing_reported)

# Evaluate each training result: 
