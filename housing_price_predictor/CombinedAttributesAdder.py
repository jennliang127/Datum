# © 2024 Jennifer Liang, jennliang127@gmail.com

import numpy as np 
from sklearn.base import BaseEstimator

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator):
    """
    This is a custom transformer that append additional attributes to 
    dataframe which assit ML training regression 
    
    Input: 
        __<function>__: generate get_params() and set_params() for convinience 
                        of hyperparameter tuning
    """
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:,population_ix]/X[:, households_ix]
        if self.add_bedrooms_per_room: 
            bedrooms_per_room = X[:,bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
if __name__== "__main__":
    attr_adder= CombinedAttributesAdder(add_bedrooms_per_room = False)
    
