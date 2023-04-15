import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


from src.exception import CustomException
from src.logger import logging
import os
import sys

from src.components.data_transformation import DataTransformation

class ModelTrainerConfig:
    def __init__(self):
        self.trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config_obj = ModelTrainerConfig()

    pass


