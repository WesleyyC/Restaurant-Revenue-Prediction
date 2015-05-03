
library(Boruta)
library(ggplot2)
library(lubridate)
library(randomForest)
library(readr)

set.seed(1)

train <- read_csv("str_num_train.csv")
train <- train[c(-17,-76,-100),]
test  <- read_csv("str_num_test.csv")

features <- c(names(train)[c(-1,-42)])

train$Revenue <- train$revenue / 1e6
train$LogRevenue <- log(train$revenue)

boruta <- Boruta(train[,features], train$LogRevenue,maxRuns = 1000)

important_features <- features[boruta$finalDecision!="Rejected"]
rf <- randomForest(train[,important_features], train$LogRevenue,ntree=1000,nPerm=1.1, importance=TRUE)

submission <- data.frame(Id=test$Id)
submission$Prediction <- exp(predict(rf, test[,important_features]))
