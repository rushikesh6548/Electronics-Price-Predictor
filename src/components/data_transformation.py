import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig():
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')


class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_obj(self):  # Will return a data transformation object and will save it as well as a pickle file

        try:
            categorical_cols_names  = ['brand_name', 'model_name', 'processor_brand', 'processor_model','ram_type', 'operating_sys']
            numerical_cols_name = ['ram_size', 'ssd_size_gb', 'hdd_size_gb', 'display_size', 'warranty','touchscreen', 'graphics_size']

            transformation = ColumnTransformer(transformers = [
                ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'),categorical_cols_names),
                ('st_scaler', StandardScaler(), numerical_cols_name)
            ])

            # We will also save this fie as a .pkl file in artifacts folder:
            # E:\TUTS (Code)\Python\Project\ElectronicsPricePredictionEndToEnd\src\components\artifact

            save_dir = os.path.join(os.getcwd(),'E:\TUTS (Code)' ,'Python','Project','ElectronicsPricePredictionEndToEnd','src', 'components', 'artifact')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(os.path.join(save_dir, 'transformation.pkl'), 'wb') as f:
                pickle.dump(transformation, f)

            return transformation # This is the preprocessor object that will be returned so that when called it will be applied to the data


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Initiated data transformation ")

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read Train and test data and saved it as pandas Dataframe")

            logging.info("Obtaining Transformation object and saving a .pkl file ")

            transformation_object = self.get_transformation_obj()

            logging.info("Loaded transformation object and saved a transformation.pkl file in artifacts ")

            target_column_name = 'price'

            input_features_train_df = train_df.drop([target_column_name], axis=1)  # This is our input features for training data !  [Independent Variables]
            target_feature_train_df = train_df[target_column_name] # This is our training data's target column i.e , price  [Dependent Variables]
            # We applied a log as we saw in EDA that the price column is higly right skewed !

            input_features_test_df = test_df.drop([target_column_name], axis = 1) # These are our input features for testing data [Independent variables]
            target_feature_test_df = test_df[target_column_name] # This is our testing data's target column i.e , price [Dependent Variable]

            logging.info("Applying Transformation object i.e , One hot Encoding the categorical columns and Standard Scaling the Numerical Columns")

            categorical_cols_names = ['brand_name', 'model_name', 'processor_brand', 'processor_model', 'ram_type',
                                      'operating_sys']
            numerical_cols_name = ['ram_size', 'ssd_size_gb', 'hdd_size_gb', 'display_size', 'warranty', 'touchscreen',
                                   'graphics_size']

            input_features_train_processed_arr = transformation_object.fit_transform(input_features_train_df)
            input_features_test_processed_arr = transformation_object.transform(input_features_test_df)

            # train_arr = np.concatenate(input_features_train_processed_arr, np.array(target_feature_train_df))  # Our final train data that will be sent to our models
            #
            # test_arr = np.concatenate(input_features_test_processed_arr, np.array(target_feature_test_df))  # Our final train data that will be sent to our models

            logging.info("PERFORMED DATA TRANSFORMATION OF TRAIN AND TEST DATA INPUT FEATURES")

            logging.info("RETURNED THE TRAIN , TEST Data INPUT FEATURES WITH INITIATE DATA TRANSFORMATION")

            return (
                input_features_train_processed_arr,
                input_features_test_processed_arr,
                target_feature_train_df,
                target_feature_test_df


            )

        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    datatrans_obj = DataTransformation()
    datatrans_obj.get_transformation_obj()
    datatrans_obj.initiate_data_transformation(train_data_path=r'E:\TUTS (Code)\Python\Project\ElectronicsPricePredictionEndToEnd\src\components\artifact\train.csv',test_data_path=r'E:\TUTS (Code)\Python\Project\ElectronicsPricePredictionEndToEnd\src\components\artifact\test.csv')

