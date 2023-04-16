from src.exception import CustomException
from src.logger import logging
import os
import sys
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split





class DataIngestionConfig:
    def __init__(self):
        self.train_data_path : str = os.path.join('artifact','train.csv')
        self.test_data_path : str = os.path.join('artifact','test.csv')
        self.raw_data_path : str = os.path.join('artifact','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion part ")
        try:  # For this one we are reading the data from our local files system that we scraped and saved locally
            df = pd.read_excel(r'E:\TUTS (Code)\Python\Project\ElectronicsPricePredictionEndToEnd\src\data\main_data\final_laptop_Cleaned.xlsx')
            logging.info("Read the data as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False , header = True)

            logging.info("Train test split initiated")

            train_data , test_data = train_test_split(df,test_size=0.15,random_state=2)

            train_data.to_csv(self.ingestion_config.train_data_path,index = False, header = True)

            test_data.to_csv(self.ingestion_config.test_data_path , index = False , header = True)

            logging.info("Train and test data split done and saved to path")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e :
            raise CustomException(e,sys)


if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    ingestion_obj.initiate_data_ingestion()





