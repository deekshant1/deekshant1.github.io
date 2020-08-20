---
title: "Practical Machine Learning Project"
author: "Deekshant"
date: "19/08/2020"
output: 
  html_document:
    keep_md: true
---




## Overview

The goal of this project is to predict the manner in which 6 participants performed some exercises. The exercises are in the "classe" variable of the data (From A to E). The machine learning algorithm described here is applied to the 20 test cases available in the validation data. This project is for answering the questions asked in the quiz based on this machine learning algorithm. We have used data from accelerometers on the belt, forearm, arm, and dumbell.
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions

Read more: http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz6VYvuV0KL (see the section on the Weight Lifting Exercise Dataset).

They have been very generous in providing us with the dataset.


## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data Processing

We will load the required libraries required for the model prediction.


```r
library(caret)
library(randomForest)
library(rattle)
library(rpart)
library(rpart.plot)
library(corrplot)
```

## Getting and Cleaning Data

We'll download the dataset and clean accordingly to get the required subset of data for analysis.


```r
if (!file.exists("pmltrain.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "H:/My R Projects/8.Practical Machine Learning/pmltrain.csv")}

if (!file.exists("pmltest.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "H:/My R Projects/8.Practical Machine Learning/pmltest.csv")
}
```

Read the train and test csv files.


```r
trainset <- read.csv("pmltrain.csv")
validation <- read.csv("pmltest.csv")
```

As this data also contains many variables with NA values. So we will now subset the required data and remove variables which have NA values in all observations.


```r
traindata <- trainset[, colSums(is.na(trainset))==0]
validdata <- validation[, colSums(is.na(validation))==0]
```



```r
dim(traindata)
```

```
## [1] 19622    93
```

```r
dim(validdata)
```

```
## [1] 20 60
```

We see that the variables of traindata are reduced to 93 and of testdata are reduced to 60.  

## Cross validation

We will now divide our traindata into training and testing set and later we will use our validation set to check about accuracy of our selected model.


```r
inTrain <- createDataPartition(traindata$classe, p = 0.7, list = FALSE)
training <- traindata[inTrain,]
testing <- traindata[-inTrain, ]

dim(training)
```

```
## [1] 13737    93
```

```r
dim(testing)
```

```
## [1] 5885   93
```

We will now remove the variables which have near to zero variance as they will have negligible impact in outcome of `classe` variable.


```r
nzv <- nearZeroVar(training)
training <- training[, -nzv]
testing <- testing[, -nzv]

nzvvalid <- nearZeroVar(validdata)
validdata <- validdata[, -nzvvalid]
```

We will now remove first six more variables namely X (which is just the index value), user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window as they have negligible or no effect in outcome of classe.


```r
training <- training[, -c(1:6)]
testing <- testing[, -c(1:6)]
validdata <- validdata[, -c(1:6)]

dim(training)
```

```
## [1] 13737    53
```

```r
dim(testing)
```

```
## [1] 5885   53
```

```r
dim(validdata)
```

```
## [1] 20 53
```

With the above process our data is more clean and now only includes 53 important variable for our prediction.

## Correlation

Now we will like to discover the correlation between the variables. 

```r
corr <- cor(training[, -53])
corrplot(corr, method = "color", type = "lower")
```

![](project_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

From the above plot we see that the dark coloured variables are the most correlated predictor variables.

## Prediction Modelling

Now we will try to build our prediction model by exploring different model types and how well they fit to our data and which provides us with the best accuracy over our data.  

We will build the following three models and explore their accuracy to determine which model to select to predict the classe variable.

1. Rpart (or Classification Tree)
2. Random Forest
3. Generalized Boosted Model

### Rpart Model

We will first build classification tree using the method `rpart` for our first prediction model. We will then use prp() funtion to plot our classification tree as dendogram.





```r
set.seed(12321)
modelrpart <- rpart(classe~., data = training, method = "class")
prp(modelrpart, cex=0.45)
```

![](project_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

We will now test our Rpart model on the testing dataset to find out its accuracy by building the confusion matrix.



```r
predictrpart <- predict(modelrpart, newdata = testing, type = "class")
confmatrpart <- confusionMatrix(predictrpart, as.factor(testing$classe))
confmatrpart
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1516  232   19   84   45
##          B   49  670  170   67  228
##          C   38  145  785  141  128
##          D   65   88   52  644  104
##          E    6    4    0   28  577
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7123          
##                  95% CI : (0.7006, 0.7239)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6345          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9056   0.5882   0.7651   0.6680  0.53327
## Specificity            0.9098   0.8917   0.9070   0.9372  0.99209
## Pos Pred Value         0.7996   0.5659   0.6346   0.6758  0.93821
## Neg Pred Value         0.9604   0.9002   0.9481   0.9351  0.90417
## Prevalence             0.2845   0.1935   0.1743   0.1638  0.18386
## Detection Rate         0.2576   0.1138   0.1334   0.1094  0.09805
## Detection Prevalence   0.3222   0.2012   0.2102   0.1619  0.10450
## Balanced Accuracy      0.9077   0.7400   0.8360   0.8026  0.76268
```

We find that the accuracy of the model is good, around 0.7123.  

We will now plot our confusion matrix to have a better look at our outcome.


```r
plot(confmatrpart$table, col = confmatrpart$byClass, main = paste("RPart Model - Accuracy = ", round(confmatrpart$overall["Accuracy"], 4)))
```

![](project_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

### Random Forest Model

We will now use the `rf` or random forest method to build our prediction model.


```r
modelRF <- train(classe~., data = training, method = "rf", trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE))
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.74%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    2    1    0    1 0.001024066
## B   25 2627    5    1    0 0.011662904
## C    0   13 2373   10    0 0.009599332
## D    1    2   25 2222    2 0.013321492
## E    1    1    4    8 2511 0.005544554
```

 We find that the estimate of error rate: 0.74%.We will now test our model against the testing dataset to predict the accuracy of the model by building up the confusion matrix.


```r
predictRF <- predict(modelRF, newdata = testing)
confmatRF <- confusionMatrix(predictRF, as.factor(testing$classe))
confmatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674   10    0    0    0
##          B    0 1127    4    0    1
##          C    0    2 1018    8    0
##          D    0    0    4  955    1
##          E    0    0    0    1 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9925, 0.9964)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9922   0.9907   0.9982
## Specificity            0.9976   0.9989   0.9979   0.9990   0.9998
## Pos Pred Value         0.9941   0.9956   0.9903   0.9948   0.9991
## Neg Pred Value         1.0000   0.9975   0.9984   0.9982   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1915   0.1730   0.1623   0.1835
## Detection Prevalence   0.2862   0.1924   0.1747   0.1631   0.1837
## Balanced Accuracy      0.9988   0.9942   0.9951   0.9948   0.9990
```

```r
confmatRF$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9947324
```

We see that we get an accuracy of 0.9947 which is very good and the out of sample error is around 0.0053. This could also be due to some overfitting. We also see that even though random forest has provided a greater accuracy but it is relatively slow. We don't consider the speed of our  model in this assignment.    

We will now plot confusion matrix to get a better look at our model.


```r
plot(confmatRF$table, col = confmatRF$byClass, main = paste("Random Forest Model - Accuracy = ", round(confmatRF$overall["Accuracy"], 4)))
```

![](project_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

### Generalized Boosted Regression Model

We will now create our generalized boosted regression model using the `gbm` method.


```r
modelgbm <- train(classe~., data = training, method = "gbm", trControl = trainControl(method = "repeatedcv", number = 5, repeats = 1), verbose = FALSE)
modelgbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 52 had non-zero influence.
```

We will now test our gbm model against the testing data. We will also find the confusion matrix to find the accuracy of our gbm model.


```r
predictgbm <- predict(modelgbm, newdata = testing)
confmatgbm <- confusionMatrix(predictgbm, as.factor(testing$classe))
confmatgbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1644   37    0    3    3
##          B   22 1073   38    3   17
##          C    5   27  974   23    7
##          D    1    2   13  926   14
##          E    2    0    1    9 1041
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9614          
##                  95% CI : (0.9562, 0.9662)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9512          
##                                           
##  Mcnemar's Test P-Value : 4.749e-05       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9821   0.9421   0.9493   0.9606   0.9621
## Specificity            0.9898   0.9831   0.9872   0.9939   0.9975
## Pos Pred Value         0.9745   0.9306   0.9402   0.9686   0.9886
## Neg Pred Value         0.9929   0.9861   0.9893   0.9923   0.9915
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2794   0.1823   0.1655   0.1573   0.1769
## Detection Prevalence   0.2867   0.1959   0.1760   0.1624   0.1789
## Balanced Accuracy      0.9859   0.9626   0.9683   0.9772   0.9798
```

We see that we get an accuracy of 0.9614 which is around close to that obtained by random forest. And this model is also relatively faster than random forest.  

We will now plot our confusion matrix of our gbm model.


```r
plot(confmatgbm$table, col = confmatgbm$byClass, main = paste("GBM Model - Accuracy = ", round(confmatgbm$overall["Accuracy"], 4)))
```

![](project_files/figure-html/unnamed-chunk-19-1.png)<!-- -->

## Using the best Model

We will now use the best prediction model for our prediction of the `classe` variable against the validation data.  
We explored three models to find the one which gives the most accurate results.

1. Rpart Model: accuracy - 71.23%

2. Random Forest Model: accuracy - 99.47%

3. Generalized Boosted Model: accuracy - 96.14% 
  
We see that our Random Forest Model has given us the highest accuracy so we will use this prediction model to predict the classe on the validation data.


```r
predictvalidrf <- predict(modelRF, newdata = validdata)
predictvalidrf
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
