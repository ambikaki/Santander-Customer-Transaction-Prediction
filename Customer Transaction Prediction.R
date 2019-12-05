rm(list = ls())
setwd("C:/users/Radhika/Desktop")
getwd()
library(readr)
install.packages("glmnet")
install.packages("ROSE")
install.packages("DMwR")
install.packages("yardstick")
x = c(" tidyverse", "DataExplorer", "caret", "Matrix", "pdp", "mlbench","caTools",
      "glmnet", "mlr", "vita", "rBayesianOptimization","lightgbm", "pROC","DMwR", "ROSE",
      "yardstick")
lapply(x, require, character.only = TRUE)

test_df <- read.csv("test.csv")
train <- read.csv("train.csv")
str(train)
dim(train)
head(train) #FIRST FIVE ROWS OF DATA
sum(is.na(train)) #checking missing values
train$target<-as.factor(train$target)

require(gridExtra)
#Count of target classes
table(train$target)
#Percenatge counts of target classes
table(train$target)/length(train$target)*100

#Bar plot for count of target classes
library(ggplot2)
ggplot(train,aes(target))+theme_bw()+geom_bar(stat='count',fill='blue')

#We have a unbalanced data,where 90% of the data is the data of number of customers those will not make a transaction and 10% of the data is those who will make a transaction.

#Distribution of train attributes from 3 to 102

for (var in names(train)[c(3:102)]){
  target<-train$target
  plot<-ggplot(train, aes(x=train[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}

#Distribution of train attributes from 103 to 202

for (var in names(train)[c(103:202)]){
  target<-train$target
  plot<-ggplot(train, aes(x=train[[var]], fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}

#We can observed that their is a considerable number of features which are significantly have different distributions for two target variables. For example like var_0,var_1,var_9,var_198 var_180 etc.
#We can observed that their is a considerable number of features which are significantly have same distributions for two target variables. For example like var_3,var_7,var_10,var_171,var_185 etc.

#Correlations in train data
#convert factor to int
train$target<-as.numeric(train$target)
train_correlations<-cor(train[,c(2:202)])
train_correlations

#We can observed that the correlation between the train attributes is very small.

#Correlations in test data
test_correlations<-cor(test_df[,c(2:201)])
test_correlations

#We can observe that the correlation between the test attributes is very small.

#Feature engineering
#Variable importance

#Variable importance is used to see top features in dataset based on mean decreses gini.

#Let us build simple model to find features which are more important.

#Split the training data using simple random sampling
train_index<-sample(1:nrow(train),0.75*nrow(train))
#train data
train_data<-train[train_index,]
#validation data
valid_data<-train[-train_index,]
#dimension of train and validation data
dim(train_data)
dim(valid_data)

#### ------- Random forest classifier ---------- ###

#Training the Random forest classifier
set.seed(2732)
#convert to int to factor
train_data$target<-as.factor(train_data$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)
#fitting the ranndom forest
library(randomForest)
rf<-randomForest(target~.,train_data[,-c(1)],mtry=mtry,ntree=10,importance=TRUE)
#Feature importance by random forest
#Variable importance
VarImp<-importance(rf,type=2)
VarImp


#We can observed that the top important features are var_12, var_26, var_22,v var_174, var_198 and so on based on Mean decrease gini.

#Handling of imbalanced data

#following are  different approaches for dealing with imbalanced datasets.

#Change the performance metric
#Oversample minority class
#Undersample majority class

#start with simple Logistic regression model.

#Split the data using CreateDataPartition
set.seed(689)
#train.index<-createDataPartition(train_df$target,p=0.8,list=FALSE)
train.index<-sample(1:nrow(train),0.8*nrow(train))
#train data
train.data<-train[train.index,]
#validation data
valid.data<-train[-train.index,]
#dimension of train data
dim(train.data)
#dimension of validation data
dim(valid.data) 
#target classes in train data
table(train.data$target)
#target classes in validation data
table(valid.data$target)

###------ Logistic Regression model -----###

#Training and validation dataset

#Training dataset
X_t<-as.matrix(train.data[,-c(1,2)])
y_t<-as.matrix(train.data$target)

#validation dataset
X_v<-as.matrix(valid.data[,-c(1,2)])
y_v<-as.matrix(valid.data$target)

#test dataset
test<-as.matrix(test_df[,-c(1)])

#Logistic regression model
library(glmnet)
set.seed(667) # to reproduce results
lr_model <-glmnet(X_t,y_t, family = "binomial")
summary(lr_model)

#Cross validation prediction
set.seed(8909)
cv_lr <- cv.glmnet(X_t,y_t,family = "binomial", type.measure = "class")
cv_lr

#Plotting the missclassification error vs log(lambda) where lambda is regularization parameter
#Minimum lambda
cv_lr$lambda.min
#plot the auc score vs log(lambda)
plot(cv_lr)
#We can observed that miss classification error increases as increasing the log(Lambda)

#Model performance on validation dataset
set.seed(5363)
cv_predict.lr<-predict(cv_lr,X_v,s = "lambda.min", type = "class")
cv_predict.lr
#Accuracy of the model is not the best metric to use when evaluating the imbalanced datasets as it may be misleading. So, we are going to change the performance metric.

#Confusion matrix
set.seed(689)
#actual target variable
target<-valid.data$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.lr<-as.factor(cv_predict.lr)
confusionMatrix(data=cv_predict.lr,reference=target)

#predict the model
set.seed(763)
lr_pred<-predict(lr_model,test,type='class')
lr_pred

#Oversample minority class:

#It can be defined as adding more copies of minority class.
#It can be a good choice when we don't have a ton of data to work with.
#Drawback is that we are adding information.This may leads to overfitting and poor performance on test data.

#Undersample majority class:

#It can be defined as removing some observations of the majority class.
#It can be a good choice when we have a ton of data -think million of rows.
#Drawback is that we are removing information that may be valuable.This may leads to underfitting and poor performance on test data.
#Both Oversampling and undersampling techniques have some drawbacks. 

#Random Oversampling Examples(ROSE)

#It creates a sample of synthetic data by enlarging the features space of minority and majority class examples.

#Random Oversampling Examples(ROSE)

set.seed(699)
train.rose <- ROSE(target~., data =train.data[,-c(1)],seed=32)$data
#target classes in balanced train data
table(train.rose$target)
valid.rose <- ROSE(target~., data =valid.data[,-c(1)],seed=42)$data
#target classes in balanced valid data
table(valid.rose$target)

#Let us see how baseline logistic regression model performs on synthetic data points.
#Logistic regression model
set.seed(462)
lr_rose <-glmnet(as.matrix(train.rose),as.matrix(train.rose$target), family = "binomial")
summary(lr_rose)

#Cross validation prediction
set.seed(473)
cv_rose = cv.glmnet(as.matrix(valid.rose),as.matrix(valid.rose$target),family = "binomial", type.measure = "class")
cv_rose

#Plotting the missclassification error vs log(lambda) where lambda is regularization parameter
#Minimum lambda
cv_rose$lambda.min
#plot the auc score vs log(lambda)
plot(cv_rose)

#Model performance on validation dataset
set.seed(442)
cv_predict.rose<-predict(cv_rose,as.matrix(valid.rose),s = "lambda.min", type = "class")
cv_predict.rose

#Confusion matrix
set.seed(478)
#actual target variable
target<-valid.rose$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cv_predict.rose<-as.factor(cv_predict.rose)
#Confusion matrix
confusionMatrix(data=cv_predict.rose,reference=target)

#therefore,We can observe that ROSE model is performing well on imbalance data compare to baseline logistic regression which can be used for further purposes.

