# Practical Machine Learning Exercise #
Author: stitchx39

## Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement â a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

We are interested to predict the manner in which they did the exercise. This is the "classe" variable in the training set. The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

The training data will be split into 70% for training and 30%  for validation. Two models were evaluated, one using Decision Tree and the other using Random Forest algorithms. The model built using the Random Forest algorithms gave the highest accuracy of 99%. This model was choosen as the final model to predict on the original testing data set.

Upon evaluating the results of the two models, Random Forest algorithm gets an accuracy of 99%. The expected out-of-sample error is estimated at 1%. With this accuracy, we expect very few or none of the test samples to be missclassified. Using the model to predict on the original testing data set and it produced a perfect 20 out of 20 correct predictions.

### Environment Settings
The following environment setting will be used.

```r
library(caret)
library(rattle)
library(rpart)
library(randomForest)
set.seed(23232)
```

### Preparing the data
Both the data sets (Training Data and Test Data) are loaded and make sure that the missing values are coded correctly. Irrelevant variables and Near Zero Variance Variables were also removed. The Training Data Set was further subdivided into 7:3 for cross-validation purpose.

```r
# Read the data
pmlTrain = read.csv("pml-Training.csv", header = TRUE, na.strings = c("NA","#DIV/0!",""))
pmlTest = read.csv("pml-Testing.csv", header = TRUE, na.strings = c("NA","#DIV/0!",""))

# Some variables are irrelevant to our current project: user_name, raw_timestamp_part_1, raw_timestamp_part_,2 cvtd_timestamp, new_window, and  num_window (columns 1 to 7). We can delete these variables.
pmlTrain = pmlTrain[ ,-c(1:7)]
pmlTest = pmlTest[ ,-c(1:7)]

# Delete columns with all missing values
pmlTrain = pmlTrain[ ,colSums(is.na(pmlTrain)) == 0]
pmlTest = pmlTest[ ,colSums(is.na(pmlTest)) == 0]

# remove near zero covariates
nsv = nearZeroVar(pmlTrain, saveMetrics = T)
pmlTrain = pmlTrain[ , !nsv$nzv]
pmlTest = pmlTest[ , !nsv$nzv]

# Partitioning Training data set into two data sets, 70% for Training, 30% for Validation:
inTrain = createDataPartition(y=pmlTrain$classe, p=0.7, list=FALSE)
TrainData = pmlTrain[inTrain, ]
TestData = pmlTrain[-inTrain, ]
```

### Building the model
Two models will be tested using Decision Tree and Random Forest algorithms. The model with the highest accuracy will be chosen as our final model.

#### Prediction Model 1: Decision Tree

```r
model1 = rpart(classe ~., data=TrainData, method="class")
# Predicting:
prediction1 = predict(model1, TestData, type = "class")
# Test results on our subTesting data set:
confusionMatrix(prediction1, TestData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1486  180   38   39   15
##          B   30  637   45   42   20
##          C   95  141  766  214  159
##          D   43   84  146  500   44
##          E   20   97   31  169  844
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7193          
##                  95% CI : (0.7076, 0.7307)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6446          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8877   0.5593   0.7466  0.51867   0.7800
## Specificity            0.9354   0.9711   0.8747  0.93558   0.9340
## Pos Pred Value         0.8453   0.8230   0.5571  0.61200   0.7270
## Neg Pred Value         0.9544   0.9018   0.9424  0.90845   0.9496
## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
## Detection Rate         0.2525   0.1082   0.1302  0.08496   0.1434
## Detection Prevalence   0.2987   0.1315   0.2336  0.13883   0.1973
## Balanced Accuracy      0.9116   0.7652   0.8106  0.72713   0.8570
```

#### Prediction Model 2: Random Forest

```r
model2 = randomForest(classe ~. , data=TrainData, method="class")
# Predicting:
prediction2 = predict(model2, TestData, type = "class")
# Test results on subTesting data set:
confusionMatrix(prediction2, TestData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    8    0    0    0
##          B    2 1130    2    0    0
##          C    0    1 1024    8    2
##          D    0    0    0  955    5
##          E    1    0    0    1 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9921   0.9981   0.9907   0.9935
## Specificity            0.9981   0.9992   0.9977   0.9990   0.9996
## Pos Pred Value         0.9952   0.9965   0.9894   0.9948   0.9981
## Neg Pred Value         0.9993   0.9981   0.9996   0.9982   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1920   0.1740   0.1623   0.1827
## Detection Prevalence   0.2853   0.1927   0.1759   0.1631   0.1830
## Balanced Accuracy      0.9982   0.9956   0.9979   0.9948   0.9966
```

The accuracy of the model using Random Forest algorithm provides the highest accuracy which is 0.99 and is selected as the final model. The expected out-of-sample error is estimated at 0.01. The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set. With an accuracy of above 99% and Testing Data set of 20 cases, we can expect very few or none of the test samples to be missclassified.

### Predictions For Testing Data
Using the Final model to predict for the 20 observations in the Testing Data set. The predicted results were submitted and were all correct.


```r
# predict outcome levels on the original Testing data set using Random Forest algorithm
predictfinal = predict(model2, pmlTest, type="class")

# Write files for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictfinal)
```
