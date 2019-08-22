pacman::p_load(readr,h2o, rstudioapi, caret, ggplot2, mlbench, lattice, C50, rpart, rpart.plot, class, dplyr, reshape2, scales, plotly, corrplot, mice, lubridate, caTools, tictoc, kknn, ranger, doSNOW, parallel,e1071)

current_path = rstudioapi::getActiveDocumentContext()$path #save working directory
setwd(dirname(current_path))
setwd("..")


#-------------------------------------------------

# Functions.

normalize <- function(M) {
  mins = t(apply(M, 2, min))
  maxs = t(apply(M, 2, max))
  ranges = maxs-mins
  Xnorm = t(apply(M,1,function(x){ (x-mins)/(ranges) }))
  Xnorm
}

standardize <- function(M) {
  means = t(apply(M,2,mean))
  sds = t(apply(M,2,sd))
  Xnorm = t(apply(M,1,function(x) {(x-means)/sds}))
  Xnorm
}


get_mode <- function(v) {
  z = sort(table(v), decreasing=T)[1]
  mode = names(which.max(z))[1]
  return(mode)
}


get_k_nearest <- function(query, examples, k, target_column_index) {
  X = examples[,-target_column_index]
  y = examples[,target_column_index]
  #calculate distances between the query and each example in the examples
  dists = apply(X, 1, function(x) { sqrt(sum((x-query)^2)) })
  # sort the distances and get the indices of this sorting
  sorted_dists = sort(dists, index.return = T)$ix
  # choose indices of k nearest
  k_nearest = sorted_dists[1:k]
  candidates = y[k_nearest]
  # get the most frequent answer
  result = get_mode(candidates)
  return(result)
}


get_weighted <- function(labels,weights) {
  dset = data.frame(labels=factor(labels), 
                    weights=weights)
  scores = aggregate(. ~ labels, dset, sum)
  result = as.character(scores$labels[which.max(scores$weights)])
  return(result)
}
s = as.numeric(c(0.1,0.2,0.1,0.4))
l = c("d","c","c","m")
get_weighted(l,s)


get_weighted_k_nearest <- function(query, examples, k, target_column_index) {
  X = examples[,-target_column_index]
  y = examples[,target_column_index]
  #calculate distances between the query and each example in the examples
  dists = apply(X, 1, function(x) { sum((x-query)^2) })
  if (any(dists==0)) {
    ind = which(dists==0)
    res = get_mode(y[ind])
    return(res)
  }
  else {
    proximities = 1/dists
    # sort the distances and get the indices of this sorting
    sorted_dists = sort(dists, index.return = T)$ix
    # choose indices of k nearest
    k_nearest = sorted_dists[1:k]
    candidates = y[k_nearest]
    weights = proximities[k_nearest]
    
    # get the most frequent answer
    result = get_weighted(candidates,weights)
    return(result)
  }
}


library(caTools)
set.seed(3711)
split = sample.split(training_data$WAP001, SplitRatio = 0.7)
train = subset(training_data, split==T)
test = subset(training_data, split==F)
testX = test[,1:520]
testy = test[,525]
tst = testX[1,]


run_experiment <- function(dataset, target_column_index, k, train_ratio, knnf, normz, n_times=1) {
  accuracy = 0
  for (i in 1:n_times) {
    # split the dataset
    target_column = dataset[,target_column_index]
    split = sample.split(target_column, SplitRatio = train_ratio)
    train = subset(dataset, split==T)
    trainX = train[,-target_column_index]
    trainy = train[,target_column_index]
    test = subset(dataset, split==F)
    testX = test[,-target_column_index]
    testy = test[,target_column_index]
    trainX = data.frame(normz(trainX))
    testX = data.frame(normz(testX))
    train = cbind(trainX,trainy)
    predictions = apply(testX, 1, knnf, train, k, ncol(train))
    dif = predictions == testy
    accuracy = accuracy + sum(dif)/length(predictions)
  }
  accuracy = accuracy/n_times
  accuracy
}


run_experiment(training_1, 521, 13, 0.8, get_weighted_k_nearest,identity)


#-------------------------------------------------


# Carga del Dataset.

training_data <- read.csv("./Datasets/trainingData.csv", sep=",")
test_data<- read.csv("./Datasets/validationData.csv", sep=",")


# Change columns format ####

# training_data$TIMESTAMP <- as.POSIXct(training_data$TIMESTAMP, origin = "1970-01-01", tz="Europe/Paris")

# training_data$FLOOR <- as.factor(training_data$FLOOR)

plot_ly(x=training_data$LATITUDE, y=training_data$LONGITUDE, z=training_data$FLOOR, type="scatter3d", mode="markers", size=3, color=training_data$FLOOR)

training_data_2 <- training_data

training_data_2 <- transform(training_data_2,BUILDING_FLOOR=paste0(BUILDINGID,FLOOR))

training_data_2 <- transform(training_data_2,SPACE_RELATIVEPOSITION=paste0(SPACEID,RELATIVEPOSITION))

plot_ly(x=test_data$LATITUDE, y=test_data$LONGITUDE, z=test_data$FLOOR, type="scatter3d", mode="markers", size=3, color=test_data$FLOOR)

test_data_2 <- test_data

test_data_2 <- transform(test_data_2,BUILDING_FLOOR=paste0(BUILDINGID,FLOOR))

test_data_2 <- transform(test_data_2,SPACE_RELATIVEPOSITION=paste0(SPACEID,RELATIVEPOSITION))


# Training and Test Visualization ####

training_data_visual <- training_data_2 [,c(521:523)]

training_data_visual$Tipo <- c(1)

test_data_visual <- test_data_2 [,c(521:523)]

test_data_visual$Tipo <- c(2)

whole_data <- rbind(training_data_visual,test_data_visual)

plot_ly(x=whole_data$LATITUDE, y=whole_data$LONGITUDE, z=whole_data$FLOOR, type="scatter3d", mode="markers", size=3, color=whole_data$Tipo)



# Knn Algorithm ####

training_data_knn <- training_data[,1:520]-100

for (i in 1:nrow(training_data_knn)) {
  for (j in 1:ncol(training_data_knn)) {
    if (training_data_knn[i,j]!=0) {
          training_data_knn[i,j] = ((-1)/training_data_knn[i,j])
    }
    else {
      training_data_knn[i,j] = training_data_knn[i,j] * 0
    }
    }}

training_data_knn$BUILDING_FLOOR <- training_data_2[,530]

training_1 <- training_data_knn



testing_data_knn <- test_data[,1:520]-100

for (i in 1:nrow(testing_data_knn)) {
  for (j in 1:ncol(testing_data_knn)) {
    if (testing_data_knn[i,j]!=0) {
      testing_data_knn[i,j] = ((-1)/testing_data_knn[i,j])
    }
    else {
      testing_data_knn[i,j] = testing_data_knn[i,j] * 0
    }
  }}

testing_data_knn$BUILDING_FLOOR <- test_data_2[,530]

testing_1 <- testing_data_knn


# Se divide el data set por edificio.####

# Solo se seleccionan las filas deseadas del datase mediante which.

training_data_b0 <- training_data_knn[which(training_data_knn$BUILDING_FLOOR=="00"| training_data_knn$BUILDING_FLOOR=="01" | training_data_knn$BUILDING_FLOOR=="02" | training_data_knn$BUILDING_FLOOR=="03"),]

training_data_b1 <- training_data_knn[which(training_data_knn$BUILDING_FLOOR=="10"| training_data_knn$BUILDING_FLOOR=="11" | training_data_knn$BUILDING_FLOOR=="12" | training_data_knn$BUILDING_FLOOR=="13"),]

training_data_b2 <- training_data_knn[which(training_data_knn$BUILDING_FLOOR=="20"| training_data_knn$BUILDING_FLOOR=="21" | training_data_knn$BUILDING_FLOOR=="22" | training_data_knn$BUILDING_FLOOR=="23" | training_data_knn$BUILDING_FLOOR=="24"),]


testing_data_b0 <- testing_data_knn[which(testing_data_knn$BUILDING_FLOOR=="00"| testing_data_knn$BUILDING_FLOOR=="01" | testing_data_knn$BUILDING_FLOOR=="02" | testing_data_knn$BUILDING_FLOOR=="03"),]

testing_data_b1 <- testing_data_knn[which(testing_data_knn$BUILDING_FLOOR=="10"| testing_data_knn$BUILDING_FLOOR=="11" | testing_data_knn$BUILDING_FLOOR=="12" | testing_data_knn$BUILDING_FLOOR=="13"),]

testing_data_b2 <- testing_data_knn[which(testing_data_knn$BUILDING_FLOOR=="20"| testing_data_knn$BUILDING_FLOOR=="21" | testing_data_knn$BUILDING_FLOOR=="22" | testing_data_knn$BUILDING_FLOOR=="23" | testing_data_knn$BUILDING_FLOOR=="24"),]


# Droplevesl to avoid the detection of Building_Floor of others data sets in this one.

training_data_b0 <- droplevels(training_data_b0)

training_data_b1 <- droplevels(training_data_b1)

training_data_b2 <- droplevels(training_data_b2)

testing_data_b0 <- droplevels(testing_data_b0)

testing_data_b1 <- droplevels(testing_data_b1)

testing_data_b2 <- droplevels(testing_data_b2)


# KNN for building ZERO.----------------------------------------------------------------------------------

# Extract 521th column of train dataset because it will be used as 'cl' argument in knn function.

building_floor_knn_train_b0 <- training_data_b0[,521]


# Extract 5th column of testing_knn dataset to measure the accuracy.

building_floor_test_b0 <- testing_data_b0[,521]


# Run knn function with k = 9. ####

knn_b0_9 <- knn(training_data_b0[,c(-521)],testing_data_b0[,c(-521)],cl=building_floor_knn_train_b0,k=6)

# Create confusion matrix.

knn_confusion_b0 <- table(knn_b0_9,building_floor_test_b0)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b0)


# LET'S MAKE A BIGGER DATA SET AND THEN SPLIT IT IN TRAINING AND TEST.

data_whole_b0 <- rbind(training_data_b0, testing_data_b0)

data_whole_b0 <- data_whole_b0[sample(nrow(data_whole_b0)),]

data_whole_b0 <- data_whole_b0[sample(nrow(data_whole_b0)),]

# Generate a random number that is 80% of the total number of rows in dataset.

inTrain2 <- sample(1:nrow(data_whole_b0), 0.8 * nrow(data_whole_b0))

# Extract training set.

data_whole_b0_training <- data_whole_b0[inTrain2,]

##extract testing set

data_whole_b0_testing <- data_whole_b0[-inTrain2,]


# Extract 521th column of train dataset because it will be used as 'cl' argument in knn function.

building_floor_knn_train_b0_whole <- data_whole_b0_training[,521]


# Extract 5th column of testing_knn dataset to measure the accuracy.

building_floor_test_b0_whole <- data_whole_b0_testing[,521]


# Run knn function with k = 9. ####

knn_b0_whole <- knn(data_whole_b0_training[,c(-521)],data_whole_b0_testing[,c(-521)],cl=building_floor_knn_train_b0_whole,k=5)

# Create confusion matrix.

knn_confusion_b0_whole <- table(knn_b0_whole,building_floor_test_b0_whole)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b0_whole)





# KNN for building ONE.------------------------------------------------------------------------------

# Extract 521th column of train dataset because it will be used as 'cl' argument in knn function.

building_floor_knn_train_b1 <- training_data_b1[,521]


# Extract 5th column of testing_knn dataset to measure the accuracy.

building_floor_test_b1 <- testing_data_b1[,521]


# Run knn function with k = 9. ####

knn_b1_9 <- knn(training_data_b1[,c(-521)],testing_data_b1[,c(-521)],cl=building_floor_knn_train_b1,k=7)

# Create confusion matrix.

knn_confusion_b1 <- table(knn_b1_9,building_floor_test_b1)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b1)


# LET'S MAKE A BIGGER DATA SET AND THEN SPLIT IT IN TRAINING AND TEST.

data_whole_b1 <- rbind(training_data_b1, testing_data_b1)

data_whole_b1 <- data_whole_b1[sample(nrow(data_whole_b1)),]

data_whole_b1 <- data_whole_b1[sample(nrow(data_whole_b1)),]

# Generate a random number that is 80% of the total number of rows in dataset.

inTrain <- sample(1:nrow(data_whole_b1), 0.8 * nrow(data_whole_b1))

# Extract training set.

data_whole_b1_training <- data_whole_b1[inTrain,]

##extract testing set

data_whole_b1_testing <- data_whole_b1[-inTrain,]


# Extract 521th column of train dataset because it will be used as 'cl' argument in knn function.

building_floor_knn_train_b1_whole <- data_whole_b1_training[,521]


# Extract 5th column of testing_knn dataset to measure the accuracy.

building_floor_test_b1_whole <- data_whole_b1_testing[,521]


# Run knn function with k = 9. ####

knn_b1_whole <- knn(data_whole_b1_training[,c(-521)],data_whole_b1_testing[,c(-521)],cl=building_floor_knn_train_b1_whole,k=3)

# Create confusion matrix.

knn_confusion_b1_whole <- table(knn_b1_whole,building_floor_test_b1_whole)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b1_whole)



# KNN for building TWO.-----------------------------------------------------------------------------

# Extract 521th column of train dataset because it will be used as 'cl' argument in knn function.

building_floor_knn_train_b2 <- training_data_b2[,521]


# Extract 5th column of testing_knn dataset to measure the accuracy.

building_floor_test_b2 <- testing_data_b2[,521]


# Run knn function with k = 9. ####

knn_b2_9 <- knn(training_data_b2,testing_data_b2,cl=building_floor_knn_train_b2,k=9)

# Create confusion matrix.

knn_confusion_b2 <- table(knn_b2_9,building_floor_test_b2)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b2)


# LET'S MAKE A BIGGER DATA SET AND THEN SPLIT IT IN TRAINING AND TEST.

data_whole_b2 <- rbind(training_data_b2, testing_data_b2)

data_whole_b2 <- data_whole_b2[sample(nrow(data_whole_b2)),]

data_whole_b2 <- data_whole_b2[sample(nrow(data_whole_b2)),]

# Generate a random number that is 80% of the total number of rows in dataset.

inTrain3 <- sample(1:nrow(data_whole_b2), 0.8 * nrow(data_whole_b2))

# Extract training set.

data_whole_b2_training <- data_whole_b2[inTrain3,]

##extract testing set

data_whole_b2_testing <- data_whole_b2[-inTrain3,]


# Extract 521th column of train dataset because it will be used as 'cl' argument in knn function.

building_floor_knn_train_b2_whole <- data_whole_b2_training[,521]


# Extract 5th column of testing_knn dataset to measure the accuracy.

building_floor_test_b2_whole <- data_whole_b2_testing[,521]


# Run knn function with k = 9. ####

knn_b2_whole <- knn(data_whole_b2_training[,c(-521)],data_whole_b2_testing[,c(-521)],cl=building_floor_knn_train_b2_whole,k=5)

# Create confusion matrix.

knn_confusion_b2_whole <- table(knn_b2_whole,building_floor_test_b2_whole)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b2_whole)




# KKNN MODEL. ####---------------------------------------------------------------------------

set.seed(123)

# Tune the cross-validation

trctrl_kknn <- trainControl(method = 'repeatedcv', number = 5, repeats = 1)

# Tune kknn parameteres


tuneGrid_kknn <- expand.grid(kmax = 1:7,            # allows to test a range of k values
                        distance = 1,        # allows to test a range of distance values
                        kernel = "optimal")

kknn_fit_b_f_b0 <- train(BUILDING_FLOOR ~ ., 
                  data = training_data_b0, 
                  method = "kknn",
                  trControl = trctrl_kknn,
                  # preProcess = c("center", "scale"),
                  tuneGrid = tuneGrid_kknn,
                  tuneLength = 7)


#------------------------------------------------------------------------------------------

tuneGrid_kknn <- expand.grid(k= c(1,2,3,4,5,6,7),            # allows to test a range of k values
                             distance = 1,        # allows to test a range of distance values
                             kernel = "optimal")

my_knn_model <- train(BUILDING_FLOOR~ .,
                      method = "knn",
                      data = training_data_b0,
                      tuneGrid = expand.grid(k= c(1,2,3,4,5,6,7)))


my_knn_model <- train(brand ~ .,
                      method = "knn",
                      data = training4,
                      tuneGrid = expand.grid(k = c(5, 9, 11, 15)))


# Intento kknn----------------------------------------------------------------------------

kknn_b0 <- kknn(BUILDING_FLOOR~., training_data_b0, testing_data_b0, distance = 1,
                  kernel = "triangular")

summary(kknn_b0)

fit_b0 <- fitted(kknn_b0)

table(testing_data_b0$BUILDING_FLOOR, fit_b0)

pcol <- as.numeric(testing_data_b0$BUILDING_FLOOR)

pairs(testing_data_b0[1:520], pch = pcol, col = c("green3", "red")
      [(testing_data_b0$BUILDING_FLOOR != fit_b0)+1])

#----------------------------------------------------------------------------------------

# kknn building 0. 

fit.train_b0 <- train.kknn(BUILDING_FLOOR ~ ., training_data_b0, kmax = 15, 
                          kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1, kcv=5)

fit.train_b0_ <- train.kknn(BUILDING_FLOOR ~ ., training_data_b0, ks = 6, 
                          kernel = "optimal", distance = 1)

table(predict(fit.train_b0, testing_data_b0), testing_data_b0$BUILDING_FLOOR)


# kknn building 1.

fit.train_b1 <- train.kknn(BUILDING_FLOOR ~ ., training_data_b1, kmax = 15, 
                           kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1)

fit.train_b1_ <- train.kknn(BUILDING_FLOOR ~ ., training_data_b1, ks = 1, 
                            kernel = "optimal", distance = 1)

table(predict(fit.train_b1, testing_data_b1), testing_data_b1$BUILDING_FLOOR)


# kknn building 2.

fit.train_b2 <- train.kknn(BUILDING_FLOOR ~ ., training_data_b2, kmax = 15, 
                           kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1)

fit.train_b2_ <- train.kknn(BUILDING_FLOOR ~ ., training_data_b2, ks = 6, 
                            kernel = "optimal", distance = 1)

table(predict(fit.train_b2, testing_data_b2), testing_data_b2$BUILDING_FLOOR)


# -----------------------------------------------------------------------------------------------


# H2O

h2o.init(nthreads = -1)

training_data_b0_h2o <- as.h2o(training_data_b0)

testing_data_b0_h2o <- as.h2o(testing_data_b0)



# -----------------------------------------------------------------------------------------------

# Preparing de data set for predict the longitud and latitud.

training_data_knn_2 <- training_data_knn

training_data_knn_2$LONGITUDE <- training_data_2[,521]

training_data_knn_2$LATITUDE <- training_data_2[,522]


testing_data_knn_2 <- testing_data_knn

testing_data_knn_2$LONGITUDE <- test_data_2[,521]

testing_data_knn_2$LATITUDE <- test_data_2[,522]


# Se divide el data set por edificio.####

# Solo se seleccionan las filas deseadas del datase mediante which.

training_data_2_b0 <- training_data_knn_2[which(training_data_knn_2$BUILDING_FLOOR=="00"| training_data_knn_2$BUILDING_FLOOR=="01" | training_data_knn_2$BUILDING_FLOOR=="02" | training_data_knn_2$BUILDING_FLOOR=="03"),]

training_data_2_b1 <- training_data_knn_2[which(training_data_knn_2$BUILDING_FLOOR=="10"| training_data_knn_2$BUILDING_FLOOR=="11" | training_data_knn_2$BUILDING_FLOOR=="12" | training_data_knn_2$BUILDING_FLOOR=="13"),]

training_data_2_b2 <- training_data_knn_2[which(training_data_knn_2$BUILDING_FLOOR=="20"| training_data_knn_2$BUILDING_FLOOR=="21" | training_data_knn_2$BUILDING_FLOOR=="22" | training_data_knn_2$BUILDING_FLOOR=="23" | training_data_knn_2$BUILDING_FLOOR=="24"),]


testing_data_2_b0 <- testing_data_knn_2[which(testing_data_knn_2$BUILDING_FLOOR=="00"| testing_data_knn_2$BUILDING_FLOOR=="01" | testing_data_knn_2$BUILDING_FLOOR=="02" | testing_data_knn_2$BUILDING_FLOOR=="03"),]

testing_data_2_b1 <- testing_data_knn_2[which(testing_data_knn_2$BUILDING_FLOOR=="10"| testing_data_knn_2$BUILDING_FLOOR=="11" | testing_data_knn_2$BUILDING_FLOOR=="12" | testing_data_knn_2$BUILDING_FLOOR=="13"),]

testing_data_2_b2 <- testing_data_knn_2[which(testing_data_knn_2$BUILDING_FLOOR=="20"| testing_data_knn_2$BUILDING_FLOOR=="21" | testing_data_knn_2$BUILDING_FLOOR=="22" | testing_data_knn_2$BUILDING_FLOOR=="23" | testing_data_knn_2$BUILDING_FLOOR=="24"),]


# Droplevesl to avoid the detection of Building_Floor of others data sets in this one.

training_data_2_b0 <- droplevels(training_data_2_b0)

training_data_2_b1 <- droplevels(training_data_2_b1)

training_data_2_b2 <- droplevels(training_data_2_b2)

testing_data_2_b0 <- droplevels(testing_data_2_b0)

testing_data_2_b1 <- droplevels(testing_data_2_b1)

testing_data_2_b2 <- droplevels(testing_data_2_b2)



# KNN for building ZERO - LONGITUDE-------------------------------------------------------------

# Extract 522th column of train dataset because it will be used as 'cl' argument in knn function.

longitude_knn_train_b0 <- training_data_2_b0[,522]


# Extract 522th column of testing_knn dataset to measure the accuracy.

longitude_test_b0 <- testing_data_2_b0[,522]


# Run knn function with k = 6. ####

knn_b0_longitude<- knn(training_data_2_b0[,-c()],testing_data_b0,cl=building_floor_knn_train_b0,k=6)

# Create confusion matrix.

knn_confusion_b0 <- table(knn_b0_9,building_floor_test_b0)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b0)


# LET'S MAKE A BIGGER DATA SET AND THEN SPLIT IT IN TRAINING AND TEST.

data_whole_longitude_b0 <- rbind(training_data_2_b0, testing_data_2_b0)

data_whole_longitude_b0 <- data_whole_longitude_b0[sample(nrow(data_whole_longitude_b0)),]

data_whole_longitude_b0 <- data_whole_longitude_b0[sample(nrow(data_whole_longitude_b0)),]

# Generate a random number that is 80% of the total number of rows in dataset.

inTrain4 <- sample(1:nrow(data_whole_longitude_b0), 0.8 * nrow(data_whole_longitude_b0))

# Extract training set.

data_whole_longitude_b0_training <- data_whole_longitude_b0[inTrain4,]

##extract testing set

data_whole_longitude_b0_testing <- data_whole_longitude_b0[-inTrain4,]


# Extract 522th column of train dataset because it will be used as 'cl' argument in knn function.

longitude_knn_train_b0_whole <- data_whole_longitude_b0_training[,522]


# Extract 522th column of testing_knn dataset to measure the accuracy.

longitude_knn_test_b0_whole <- data_whole_longitude_b0_testing[,522]


# Run knn function with k = 9. ####

knn_b0_whole_longitude <- knn(data_whole_longitude_b0_training[,-c(521,522,523)],data_whole_longitude_b0_testing[,-c(521,522,523)],cl=longitude_knn_train_b0_whole,k=5)

# Create confusion matrix.

knn_confusion_b0_whole_longitude <- table(knn_b0_whole_longitude,longitude_knn_test_b0_whole)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_confusion_b0_whole_longitude)


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


# NEW APPROACH ####

training_data_mentor <- training_data_knn

training_data_mentor$BUILDINGID <- training_data_2[,524]

training_data_mentor$FLOOR <- training_data_2[,523]

training_data_mentor$LONGITUDE <- training_data_2[,521]

training_data_mentor$LATITUDE <- training_data_2[,522]

training_data_mentor$BUILDINGID <- as.factor(training_data_mentor$BUILDINGID)

training_data_mentor$FLOOR <- as.factor(training_data_mentor$FLOOR)

training_data_mentor <- droplevels(training_data_mentor)


testing_data_mentor <- testing_data_knn

testing_data_mentor$BUILDINGID <- test_data_2[,524]

testing_data_mentor$FLOOR <- test_data_2[,523]

testing_data_mentor$LONGITUDE <- test_data_2[,521]

testing_data_mentor$LATITUDE <- test_data_2[,522]

testing_data_mentor$BUILDINGID <- as.factor(testing_data_mentor$BUILDINGID)

testing_data_mentor$FLOOR <- as.factor(testing_data_mentor$FLOOR)

testing_data_mentor <- droplevels(testing_data_mentor)


# Generate a random number that is 80% of the total number of rows in dataset.

inTrain4 <- sample(1:nrow(training_data_mentor), 0.8 * nrow(training_data_mentor))

# Extract training set.

training_data_mentor_training <- training_data_mentor[inTrain4,]

# Extract testing set

testing_data_mentor_training <- training_data_mentor[-inTrain4,]


# PREDICTING THE BUILDINGID.------------------------------------------------------------------------------------


# Extract 522th column of train dataset because it will be used as 'cl' argument in knn function.

training_data_mentor_training_building <- training_data_mentor_training[,522]


# Extract 522th column of testing_knn dataset to measure the accuracy.

testing_data_mentor_training_building <- testing_data_mentor_training[,522]


# Run knn function with k = 5. ####

knn_mentor_building <- knn(training_data_mentor_training[,-c(521:525)],testing_data_mentor_training[,-c(521:525)],cl=training_data_mentor_training_building,k=5)

# Create confusion matrix.

knn_mentor_building_confusion <- table(knn_mentor_building,testing_data_mentor_training_building)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_mentor_building_confusion)

kappa(knn_mentor_building_confusion)



# NOW, LET'S TRAIN THE MODEL WITH THE WHOLE TRAINING DATA SET AND THEN TEST IT AGAINST THE TESTING DATA SET.

# Extract 522th column of train dataset because it will be used as 'cl' argument in knn function.

training_data_mentor_training_building_fin <- training_data_mentor[,522]


# Extract 522th column of testing_knn dataset to measure the accuracy.

testing_data_mentor_training_building_fin <- testing_data_mentor[,522]


# PREDICTING THE BUILDINGID.


# Run knn function with k = 5. ####

knn_mentor_building_fin <- knn(training_data_mentor[,-c(521:525)],testing_data_mentor[,-c(521:525)],cl=training_data_mentor_training_building_fin,k=5)

# Create confusion matrix.

knn_mentor_building_confusion_fin <- table(knn_mentor_building_fin,testing_data_mentor_training_building_fin)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_mentor_building_confusion_fin)

kappa(knn_mentor_building_confusion_fin)



# PREDICTING FLOOR USING THE BUILDING AS A TRAINING PARAMETER.-----------------------------------------------------------------------------------


# Extract 523th column of train dataset because it will be used as 'cl' argument in knn function.

training_data_mentor_training_floor <- training_data_mentor_training[,523]


# Extract 523th column of testing_knn dataset to measure the accuracy.

testing_data_mentor_training_floor <- testing_data_mentor_training[,523]


# Run knn function with k = 5. ####

knn_mentor_floor <- knn(training_data_mentor_training[,-c(521,523,524,525)],testing_data_mentor_training[,-c(521,523,524,525)],cl=training_data_mentor_training_floor,k=5)

# Create confusion matrix.

knn_mentor_floor_confusion <- table(knn_mentor_floor,testing_data_mentor_training_floor)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_mentor_floor_confusion)

kappa(knn_mentor_floor_confusion)



# NOW, LET'S TRAIN THE MODEL WITH THE WHOLE TRAINING DATA SET AND THEN TEST IT AGAINST THE TESTING DATA SET.

# Extract 523th column of train dataset because it will be used as 'cl' argument in knn function.

training_data_mentor_training_floor_fin <- training_data_mentor[,523]


# Extract 523th column of testing_knn dataset to measure the accuracy.

testing_data_mentor_training_floor_fin <- testing_data_mentor[,523]



# PREDICTING FLOOR USING THE BUILDING AS A TRAINING PARAMETER.


# Run knn function with k = 5. ####

knn_mentor_floor_fin <- knn(training_data_mentor[,-c(521,523,524,525)],testing_data_mentor[,-c(521,523,524,525)],cl=training_data_mentor_training_floor_fin,k=5)


# Create confusion matrix.

knn_mentor_floor_confusion_fin <- table(knn_mentor_floor_fin,testing_data_mentor_training_floor_fin)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_mentor_floor_confusion_fin)

kappa(knn_mentor_floor_confusion_fin)


# PREDICTING FLOOR USING THE BUILDING AS A TRAINING PARAMETER BUT USING CARET.-----------------
# AND ACCELERATING CPU SPEED ---------------------------------------------------------------------------


# training_data_mentor_sample <- training_data_mentor[sample(nrow(training_data_mentor), 10000), ]


cluster <- makeCluster(detectCores()-2)
registerDoSNOW(cluster)

tic()

traincontrol_floor_final <- trainControl(method = c("boot"),
                           number = 5,
                           allowParallel = TRUE,
                           verboseIter = TRUE)

knn_floor_final <- train(FLOOR~ ., data=training_data_mentor[,-c(521,524,525)],
                          method='knn',
                          trControl = traincontrol_floor_final,
                          tuneGrid = expand.grid(k= c(1,3)))

toc()

print(knn_floor_final)



stopCluster(cluster)

registerDoSEQ(cluster)
rm(cluster)


saveRDS(knn_floor_final, "C:/Users/jonat/Documents/UBIQUM/GITHUB PROJECTS/Wifi_Locationing_Project/Models/knn_floor_final.rds")


# Performance Metrics of this model.

results_knn_floor_final <- predict(knn_floor_final,newdata = testing_data_mentor[,-c(521,524,525)])

postResample(pred = results_knn_floor_final, obs = testing_data_mentor$FLOOR)



# -------------------------------------------------------------------------------------------------------


# ADDING THE FLOOR PREDICTION AS A NEW TRAINING PARAMETER FOR THE NEXT TRAININGS.

results_knn <- predict(my_knn_model,newdata = testing4)

postResample(pred = results_knn, obs = testing4$brand)

knn_mentor_floor_fin_next <- training_data_mentor

floor_prediction_to_add <- predict(knn_mentor_floor_fin, training_data_mentor[,-c(521,523,524,525)])


floor_prediction_to_add <- predict(knn_floor_final,newdata = training_data_mentor[,-c(521,524,525)])

postResample(pred = floor_prediction_to_add, obs = training_data_mentor$FLOOR)





# PREDICTING LONGITUDE.-----------------------------------------------------------------------------------


# Extract 524th column of train dataset because it will be used as 'cl' argument in knn function.

training_data_mentor_training_longitude <- training_data_mentor_training[,524]


# Extract 524th column of testing_knn dataset to measure the accuracy.

testing_data_mentor_training_longitude <- testing_data_mentor_training[,524]



# Run knn function with k = 5. ####

training_data_mentor_training_2 <- training_data_mentor_training[,-c(521,524,525)]

testing_data_mentor_training_2 <- testing_data_mentor_training[,-c(521,524,525)]

knn_mentor_longitude <- knn(training_data_mentor_training_2,testing_data_mentor_training_2,cl=training_data_mentor_training_longitude,k=5)

# Create a data frame with the error.

knn_mentor_longitude_error <- (testing_data_mentor_training_longitude- as.numeric(as.character(knn_mentor_longitude)))

# Function that returns Root Mean Squared Error

rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error

mae <- function(error)
{
  mean(abs(error))
}

R2 <- function(error,original)
{ 1-(sum((error)^2)/sum((original-mean(original))^2))
}
  
R2 = 1-(sum((d)^2)/sum((original-mean(original))^2))


rmse_ejercicio <- rmse(knn_mentor_longitude_error)
mae_ejercicio <- mae(knn_mentor_longitude_error)
R2_ejercicio <- R2(knn_mentor_longitude_error,as.numeric(as.character(knn_mentor_longitude)))


cat(" MAE:", mae_ejercicio, "\n", 
    "RMSE:", rmse_ejercicio, "\n", "R-squared:", R2_ejercicio)


# NOW, LET'S TRAIN THE MODEL WITH THE WHOLE TRAINING DATA SET AND THEN TEST IT AGAINST THE TESTING DATA SET.

# Extract 524th column of train dataset because it will be used as 'cl' argument in knn function.

training_data_mentor_longitude_fin <- training_data_mentor[,524]


# Extract 524th column of testing_knn dataset to measure the accuracy.

testing_data_mentor_longitude_fin <- testing_data_mentor[,524]


# Run knn function with k = 5. ####

knn_mentor_longitude_whole <- knn(training_data_mentor[,-c(521,524,525)],testing_data_mentor[,-c(521,524,525)],cl=training_data_mentor_longitude_fin,k=5)

# Create a data frame with the error.

knn_mentor_longitude_error_whole <- (testing_data_mentor_longitude_fin- as.numeric(as.character(knn_mentor_longitude_whole)))

rmse_ejercicio <- rmse(knn_mentor_longitude_error_whole)
mae_ejercicio <- mae(knn_mentor_longitude_error_whole)
R2_ejercicio <- R2(knn_mentor_longitude_error_whole,as.numeric(as.character(knn_mentor_longitude_whole)))


cat(" MAE:", mae_ejercicio, "\n", 
    "RMSE:", rmse_ejercicio, "\n", "R-squared:", R2_ejercicio)



# Knn using CARET -----------------------------------------------------------------------------------------

cluster <- makeCluster(detectCores()-2)
registerDoSNOW(cluster)

tic()

knn_longitude_final <- train(LONGITUDE~ ., data=training_data_mentor[,-c(521,525)],
                         method='knn',
                         trControl = traincontrol_floor_final,
                         tuneGrid = expand.grid(k= c(1,3,5,7)))

toc()

print(knn_floor_final)



stopCluster(cluster)

registerDoSEQ(cluster)
rm(cluster)


# saveRDS(knn_floor_final, "C:/Users/jonat/Documents/UBIQUM/GITHUB PROJECTS/Wifi_Locationing_Project/Models/knn_floor_final.rds")


# Performance Metrics of this model.

results_knn_longitude_final <- predict(knn_longitude_final,newdata = testing_data_mentor[,-c(521,525)])

postResample(pred = results_knn_longitude_final, obs = testing_data_mentor$LONGITUDE)











# PREDICTING FLOOR USING THE BUILDING AS A TRAINING PARAMETER.----------------------------------------------


# Run knn function with k = 5. ####

knn_mentor_floor_fin <- knn(training_data_mentor[,-c(521,523,524,525)],testing_data_mentor[,-c(521,523,524,525)],cl=training_data_mentor_training_floor_fin,k=5)

#dummy_training_data_mentor <- fastDummies::dummy_cols(training_data_mentor[,-c(521,523,524,525)], remove_first_dummy = TRUE)

#dummy_testing_data_mentor <- fastDummies::dummy_cols(testing_data_mentor[,-c(521,523,524,525)], remove_first_dummy = TRUE)

#knn_mentor_floor_fin <- knn(dummy_training_data_mentor,dummy_testing_data_mentor,cl=training_data_mentor_training_floor_fin,k=5)


# Create confusion matrix.

knn_mentor_floor_confusion_fin <- table(knn_mentor_floor_fin,testing_data_mentor_training_floor_fin)

# This function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(knn_mentor_floor_confusion_fin)

kappa(knn_mentor_floor_confusion_fin)




