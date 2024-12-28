
"""
Created on Sat Nov 30 08:55:40 2024

@author: Sadegh Asghari
"""

library(rpart)          # For Decision Tree  
library(caret)         # For confusionMatrix function and others  
library(dplyr)         # For data manipulation  
library(pROC)          # For calculating AUC if needed  

# Function to evaluate the Decision Tree model  
evaluate_decision_tree <- function(X, y) {  
  set.seed(42)  # For reproducibility  
  metrics_list <- list()  

  # Create a data frame combining features and labels  
  data <- data.frame(X, y)  
  
  # Stratified K-Folds cross-validation  
  folds <- createFolds(data$y, k = 5, list = TRUE, returnTrain = TRUE)  
  
  # Initialize metric containers  
  accuracy <- numeric(length(folds))  
  precision <- numeric(length(folds))  
  recall <- numeric(length(folds))  
  f1_score <- numeric(length(folds))  
  mcc <- numeric(length(folds))  
  balanced_accuracy <- numeric(length(folds))  
  
  for (i in seq_along(folds)) {  
    train_indices <- folds[[i]]  
    test_indices <- setdiff(1:nrow(data), train_indices)  

    # Split data into train and test sets  
    train_set <- data[train_indices, ]  
    test_set <- data[test_indices, ]  

    # Fit the Decision Tree model  
    dt_model <- rpart(y ~ ., data = train_set, method = "class")  

    # Predictions  
    y_pred <- predict(dt_model, test_set, type = "class")  

    # Calculate metrics  
    confusion_mat <- confusionMatrix(as.factor(y_pred), as.factor(test_set$y), positive = levels(test_set$y)[1])  

    # Store metrics  
    accuracy[i] <- confusion_mat$overall['Accuracy']  
    precision[i] <- confusion_mat$byClass['Precision']  
    recall[i] <- confusion_mat$byClass['Recall']  
    f1_score[i] <- confusion_mat$byClass['F1']  
    mcc[i] <- mccr(confusion_mat$table)  # Custom function for MCC  
    balanced_accuracy[i] <- mean(c(precision[i], recall[i]))  # Balanced Accuracy as mean of precision and recall  
  }  
  
  # Return average metrics  
  return(data.frame(  
    mcc = mean(mcc) * 100,  
    accuracy = mean(accuracy) * 100,  
    precision = mean(precision) * 100,  
    recall = mean(recall) * 100,  
    f1_score = mean(f1_score) * 100,  
    balanced_accuracy = mean(balanced_accuracy) * 100  
  ))  
}  

# Function to calculate MCC from confusion matrix  
mccr <- function(cm) {  
  TP <- cm[1, 1]  
  TN <- cm[2, 2]  
  FP <- cm[1, 2]  
  FN <- cm[2, 1]  
  
  numerator <- (TP * TN) - (FP * FN)  
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  
  if (denominator == 0) {  
    return(0)  
  }  
  return(numerator / denominator)  
}
