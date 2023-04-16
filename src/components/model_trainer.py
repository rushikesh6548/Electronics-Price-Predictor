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
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestion, DataIngestionConfig

from src.components import data_transformation
from src.exception import CustomException
from src.logger import logging
import os
import sys



from src.components.data_transformation import  DataTransformationConfig
from src.components.data_transformation import DataTransformation


class ModelTrainerConfig:
    def __init__(self):
        self.trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config_obj = ModelTrainerConfig()
        self.best_model_object = None



    def initiate_model_trainer(self,input_features_training,input_features_testing,target_feature_training,target_feature_testing):
        try:
            X_train, y_train, X_test, y_test = (
                input_features_training,
                np.log(target_feature_training),
                input_features_testing,
                np.log(target_feature_testing)
            )

            logging.info("Defined X_train,y_train,X_test,y_Test")

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Lasso" : Lasso(),
                "KNN":KNeighborsRegressor(),
                "Random Forest": RandomForestRegressor(),
                "CatBoost":CatBoostRegressor(),
                "AdaBoost":AdaBoostRegressor()
            }


            model_report :dict = {}

            logging.info("Initiating Model Evaluation for all the models")
            def model_evaluate(X_train = X_train,y_train= y_train,X_test= X_test,y_test= y_test,model = None):  # This function will get model performance for a specific model
                try:
                    model.fit(X_train,y_train)
                    y_pred = model.predict(X_test)
                    r2score = r2_score(y_test,y_pred)
                    mae = mean_absolute_error(y_test,y_pred)
                    mse = mean_squared_error(y_test,y_pred)
                    return {'r2_score':r2score,"MSE":mse,"MAE":mae}

                except Exception as e:
                    raise CustomException(e,sys)



            for model_name , model_object in models.items():
                report = model_evaluate(X_train= X_train,y_train= y_train,X_test=X_test,y_test=y_test,model = model_object)
                model_report[model_name] = report

            logging.info("Completed Model Evaluation for all models")


            logging.info("Finding the best model ")


            # Sorting by best r2 score :
            sorted_model_report = sorted(model_report.items(), key=lambda x: x[1]['r2_score'], reverse=True)
            best_model_name = sorted_model_report[0][0]

            best_model_object = None
            best_model_r2score = None

            for model_name , model_object in models.items():
                if model_name == best_model_name:
                    best_model_object = model_object

            for model_name, report in model_report.items():
                if model_name == best_model_name:
                    best_model_r2score = report['r2_score']

            # If we dont find any model score > 70 we raise an error !
            if best_model_r2score < 0.70:
                raise CustomException("NO model found with r2 score > 0.70")

            if best_model_r2score > 0.70:
                logging.info(f"Best Model Found and it is :{best_model_name} with an r2 score of {best_model_r2score}")

            self.best_model_object = best_model_object

        except Exception as e:
            raise CustomException(e, sys)

    def save_best_model(self):
        save_dir = os.path.join(os.getcwd(), 'E:\TUTS (Code)', 'Python', 'Project',
                                'ElectronicsPricePredictionEndToEnd', 'src', 'components', 'artifact')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, 'trained_model.pkl'), 'wb') as f:
            pickle.dump(self.best_model_object, f)

        print(self.best_model_object)






ingestion_obj = DataIngestion()
ingestion_obj.initiate_data_ingestion()
datatrans_obj = DataTransformation()
input_features_training , input_features_testing , target_features_training , target_features_testing = datatrans_obj.initiate_data_transformation(train_data_path=r'E:\TUTS (Code)\Python\Project\ElectronicsPricePredictionEndToEnd\src\components\artifact\train.csv',test_data_path=r'E:\TUTS (Code)\Python\Project\ElectronicsPricePredictionEndToEnd\src\components\artifact\test.csv')
print(input_features_training.shape,input_features_testing.shape,target_features_training.shape,target_features_testing.shape)
mt = ModelTrainer()
mt.initiate_model_trainer(input_features_training , input_features_testing , target_features_training , target_features_testing)
mt.save_best_model()

