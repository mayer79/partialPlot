#======================================================================
# Examples for partialPlot
#======================================================================

library(xgboost)
library(lightgbm)
library(ranger)
source("R/partialPlot.R") # or your path


#======================================================================
# Data prep 
#======================================================================

train <- list(y = as.numeric(iris[, 5]),
              X = as.matrix(iris[, 1:4]))


#======================================================================
# xgboost
#======================================================================

dtrain <- xgb.DMatrix(train$X, label = as.numeric(train$y) - 1)

param <- list(max_depth = 2, learning_rate = 0.1, objective = "multi:softprob", 
              num_class = 3, eval_metric = "merror")

fit_xgb <- xgb.train(dtrain, params = param, nrounds = 100)

par(mfrow = c(2, 2))

for (nam in colnames(train$X)) {
  partialPlot(fit_xgb, train$X, xname = nam, xlab = "", which.class = 0)
}


#======================================================================
# lightgbm
#======================================================================

dtrain <- lgb.Dataset(train$X, 
                      label = as.numeric(train$y) - 1)

params <- list(objective = "multiclass", 
               learning_rate = 0.1,
               min_data = 20,
               metric = "multi_error", 
               num_class = 3)

system.time(fit_lgb <- lgb.train(data = dtrain,
                                 params = params, 
                                 nrounds = 100,
                                 verbose = 0L))

par(mfrow = c(2, 2))

for (nam in colnames(train$X)) {
  partialPlot(fit_lgb, train$X, xname = nam, xlab = "", which.class = 0)
}


#======================================================================
# ranger
#======================================================================

library(ranger)
source("R/partialPlot.R") # or your path

fit_ranger <- ranger(Species ~ ., data = iris, probability = TRUE)

par(mfrow = c(2, 2))

for (nam in colnames(train$X)) {
  partialPlot(fit_ranger, train$X, xname = nam, xlab = "", which.class = "setosa")
}

