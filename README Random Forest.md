Loading Libraries: The randomForest, caret, and dplyr libraries are loaded for model fitting and data manipulation.

Function Definition: The evaluate_random_forest function performs stratified K-fold cross-validation similar to your Python function. It combines features and target variables into a single data frame.

Model Fitting: A Random Forest model is fit using the randomForest() function, and predictions are generated.

Metric Calculation: Metrics such as accuracy, precision, recall, F1 score, MCC, and balanced accuracy are calculated using the confusion matrix.

Custom MCC Function: A helper function computes the Matthews Correlation Coefficient from the confusion matrix.

Output: The function returns a data frame containing the average metrics across all folds.
