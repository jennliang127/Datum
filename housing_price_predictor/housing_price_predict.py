from imports import * 
from DataPrep import DataPrep
from CombinedAttributesAdder import CombinedAttributesAdder

# read csv data as DataFrame
file_name = 'housing.csv'
read_file = os.path.join(os.getcwd(),f'data/{file_name}')
housing_raw = pd.read_csv(read_file)


# Exploratory Data Analysis (EDA): 

housing_raw.info()
housing_raw.keys()
housing_raw.head()
housing_raw.describe()

housing_raw['ocean_proximity'].value_counts() # Check the type of subject exist in the object field

housing_raw.hist(bins=50) # check given info distribution 
                          ## Note: median_income and median_house_value have 
                          #        a skewed distribution. total_rooms, tota_bedrooms,
                          #        population, households have similar distributions 

# Check if there are any null values exist in DataFrame:
housing_raw.isnull().values.any()   # Choose to fill in missing values with 
