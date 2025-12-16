House Price Prediction using Machine Learning

---> Overview:

This project builds a complete machine learning pipeline to predict housing prices based on numerical features such as size, number of bedrooms, bathrooms, and other structural attributes.
It includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization.
The goal is to demonstrate a full end-to-end machine learning workflow using real-world housing data.


 ---> Features:
 
•	Data loading & preprocessing
•	Handling missing values
•	Exploratory Data Analysis (EDA): distributions, correlations
•	Feature selection
•	Model training (Linear Regression & Random Forest)
•	Evaluation using RMSE and R²
•	Visualization of predicted vs actual values
•	Fully reproducible and extendable ML pipeline


---> Technologies Used:
•	Python
•	Pandas
•	NumPy
•	Matplotlib
•	Seaborn
•	Scikit-learn
•	Jupyter Notebook


---> Modeling Approach:
  1. Two models are trained:
    - Linear Regression a simple baseline model for comparison.
    - Random Forest Regressor a more powerful non-linear model that performs better for housing data.

  2. Evaluation metrics used:
	- MSE
	- RMSE
	- R² Score


---> Results Summary:
•	Random Forest outperforms Linear Regression on test data
•	R² score indicates how much variance in housing prices the model explains
•	Actual vs Predicted scatter plot included


---> Future Improvements:
•	Add feature importance ranking
•	Try Gradient Boosting / XGBoost
•	Hyperparameter tuning
•	Add categorical feature encoding


