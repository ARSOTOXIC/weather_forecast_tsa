# weather_forecast_tsa

STEPS---
--->Importing Libraries: Importing necessary libraries like Pandas, NumPy, Matplotlib, Seaborn, and various modules from Scikit-Learn.
--->Data Collection: Reading weather data from a CSV file using Pandas.
--->Data Exploration: Examining the dataset structure, checking for missing values, duplicates, and outliers.
--->Data Preprocessing:
--->Handling missing values in the 'Precip Type' column.
--->Removing rows with zero values in 'Humidity' and 'Pressure' columns to address outliers.
--->Correlation Analysis: Exploring correlations among different attributes.
--->Creating Training and Test Sets: StratifiedShuffleSplit used to create training and test sets.
--->Feature Engineering: Normalizing numerical attributes and encoding categorical attributes using Pipelines and Transformers.
VBuilding Models:
Linear Regression, Polynomial Regression, Decision Tree Regressor, Random Forest Regressor are trained and evaluated on the training set.
--->Hyperparameter Tuning:
Grid Search and Randomized Search to find the best hyperparameters for Random Forest Regressor.
--->Model Evaluation:
Evaluating the best models on the test set using metrics like RMSE, R2 score, and cross-validation.
--->Model Comparison: Comparing actual vs. predicted values using visualizations like bar plots and distribution plots.
--->Model Performance: Calculating the explained variance and accuracy of predictions.
Conclusion: A short note about potential improvements or feedback requested for enhancing the notebook.
This code is an end-to-end exploration and implementation of machine learning models for weather prediction using regression techniques.





