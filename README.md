# Flight Delay Prediction using Apache Spark

## Overview

This project aims to leverage the power of Apache Spark to predict flight delays using historical flight data. By applying machine learning techniques within the Spark ecosystem, we build and evaluate models capable of forecasting the extent of delays a flight might experience. This endeavor not only demonstrates the application of data processing and machine learning at scale but also provides insights that could potentially improve flight scheduling, passenger convenience, and airline operations.

## Data Processing

The data processing phase is critical in transforming raw flight data into a format suitable for machine learning models. This process involves several steps:

- **Cleaning and Preprocessing**: Initial steps involve cleaning the data by removing unnecessary columns, filtering out canceled or non-delayed flights, and dealing with missing values to ensure the quality and completeness of the dataset.

- **Feature Engineering**: We conduct feature engineering to prepare and transform variables for better predictive performance. This includes encoding categorical variables like airline carriers and airports into numerical values using target mean encoding and handling cyclical features such as dates and times to preserve their cyclical nature.

- **Normalization**: Features are normalized to ensure that they have a uniform scale, improving the stability and performance of the machine learning models.

- **Training and Test Split**: The dataset is split into training, validation, and test sets, allowing for the evaluation of model performance on unseen data, ensuring that the models generalize well to new data.

## Models

Two models are developed to predict flight delays:

- **Linear Regression**: Serves as a baseline model, providing a straightforward approach to understanding the linear relationships between the independent variables and the target variable (flight delay).

- **Random Forest Regressor**: A more complex model that can capture nonlinear relationships and interactions between variables. It's expected to perform better due to its ability to model complex patterns in the data.

Both models are evaluated based on their Root Mean Squared Error (RMSE) and R-squared values, metrics that offer insights into the accuracy and predictive power of the models.

## Results

The results section discusses the performance of the Linear Regression and Random Forest Regressor models. We present the RMSE and R-squared values obtained on the training, validation, and test datasets for each model. This comparison helps in understanding which model performs better in predicting flight delays based on historical data and sheds light on the effectiveness of the feature engineering and data preprocessing steps. Insights drawn from the model evaluations can guide future improvements and iterations of the modeling process.
