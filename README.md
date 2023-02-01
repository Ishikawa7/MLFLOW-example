# MLflow Grid Search for XGBoost Model
Usage of MLFLOW for tracking models experiments (XGBoost) and serve a containerized model.
This code performs a grid search for hyperparameter tuning of an XGBoost classifier using the MLflow library. The grid search is performed over a range of parameters (max_depth, scale_pos_weight, learning_rate, and n_estimators).

## Dependencies
The following libraries are required to run this code:

- mlflow
- numpy
- pandas
- warnings
- json
- os
- product from itertools
- XGBoost
- sklearn

## Results
The best performing hyperparameters are printed to the console, as well as their corresponding F1 score. Additionally, the results of the grid search are logged to the MLflow tracking server, allowing you to visualize and compare the results of each run. The metrics of each run are stored as JSON files and logged as artifacts to the MLflow server.

## Note
The train_xgboost_pipeline function and the data used for training and validation are included in this repository.
