# Electronics Price Predictor

This is an end-to-end machine learning project for predicting the prices of electronic devices, specifically laptops, based on scraped data from Flipkart. 
Try the app youself : https://electronics-price-predictor.streamlit.app/
A preview of the app : 
![image](https://user-images.githubusercontent.com/46419407/234588405-1858cfe5-1403-44c4-85dc-67c38d6e831f.png)


![image](https://user-images.githubusercontent.com/46419407/234588480-6c884914-3347-46e2-9819-9ff32a91acfb.png)

## Data Collection

The data used for this project was collected by scraping data from the Flipkart website using BeautifulSoup library in Python.

## Data Preprocessing and Feature Engineering

The scraped data was preprocessed and feature engineering was applied to extract relevant features that are potential indicators of the price of the electronic devices. Some of the features used in the model include the brand, the processor, the RAM, the storage, the graphics card, and the screen size.

## Data Ingestion

The preprocessed data was read and ingested into the machine learning model.

### Data Transformation

After the raw data was preprocessed and feature engineering was applied, a transformation object was created using `StandardScaler` to scale the input features. Additionally, we created categories on the prices of the electronic devices so that we could classify them. These transformations were saved as a pickle file named `transformation.pkl`, which was later applied to the data during model inference.

### Model Training

After the preprocessed data was ingested and transformed, regression models were evaluated on the data, and the best model was selected based on its performance metrics. The best model on the basis of performance came out to be `CatBoost Regressor`. The trained model was saved in a pickle file as `model_trained.pkl`.

### App Deployment

After the model was trained, a web app was created with the necessary templates using Streamlit. The app was deployed using Streamlit sharing or Heroku so that users can access it easily. 

## Conclusion

In conclusion, this project demonstrates an end-to-end machine learning pipeline for predicting the prices of electronic devices based on scraped data from Flipkart. The trained model can be used to predict the prices of new devices given their specifications, making it useful for both consumers and manufacturers.
