# © 2024 Jennifer Liang, jennliang127@gmail.com

# General imports
import os
import random
import numpy as np 
import pandas as pd
from zlib import crc32
import matplotlib.pyplot as plt 

from scipy import stats

# Import data cleaning, transformation tools
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit

# Import fine tuning tools 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt import dummy_minimize
from skopt import gp_minimize
from skopt import forest_minimize

# Improt regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Import evaluation tools 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Import customize transformation
from CombinedAttributesAdder import CombinedAttributesAdder