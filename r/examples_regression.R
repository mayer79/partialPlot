#======================================================================
# Regression Examples for "partialPlot"
#======================================================================

library(ggplot2) # for data set "diamonds"
library(xgboost)
library(lightgbm)
library(ranger)
source("R/partialPlot.R") # or your path


#======================================================================
# Data prep 
#======================================================================

diamonds <- transform(as.data.frame(diamonds),
                      log_price = log(price),
                      log_carat = log(carat),
                      cut = as.numeric(cut),
                      color = as.numeric(color),
                      clarity = as.numeric(clarity))

# Train/test split
set.seed(3928272)
.in <- sample(c(FALSE, TRUE), nrow(diamonds), replace = TRUE, p = c(0.15, 0.85))

x <- c("log_carat", "cut", "color", "clarity", "depth", "table")

train <- list(y = diamonds$log_price[.in],
              X = as.matrix(diamonds[.in, x]))
test <- list(y = diamonds$log_price[!.in],
             X = as.matrix(diamonds[!.in, x]))

trainDF <- diamonds[.in, ]
testDF <- diamonds[!.in, ]

#======================================================================
# Small functions
#======================================================================

# Calculate R squared
r2 <- function(y, pred) {
  1 - var(y - pred) / var(y)  
}

# Show all partial dependency plots
partialDiamondsPlot <- function(fit) {
  par(mfrow = 3:2,
      oma = c(0, 0, 0, 0) + 0.3,
      mar = c(4, 2, 0, 0) + 0.1,
      mgp = c(2, 0.5, 0.5))
  
  partialPlot(fit, train$X, xname = "log_carat")
  partialPlot(fit, train$X, xname = "cut", discrete.x = TRUE)
  partialPlot(fit, train$X, xname = "color", discrete.x = TRUE)
  partialPlot(fit, train$X, xname = "clarity", discrete.x = TRUE)
  partialPlot(fit, train$X, xname = "depth")
  partialPlot(fit, train$X, xname = "table")
}

#======================================================================
# xgboost regression
#======================================================================

dtrain <- xgb.DMatrix(train$X, label = train$y)
dtest <- xgb.DMatrix(test$X, label = test$y)
watchlist <- list(train = dtrain, test = dtest)

param <- list(max_depth = 8, 
              learning_rate = 0.01, 
              nthread = 2, 
              lambda = 0.2, 
              objective = "reg:linear", 
              eval_metric = "rmse", 
              subsample = 0.7)

fit_xgb <- xgb.train(param, dtrain, watchlist = watchlist, 
                     nrounds = 850, early_stopping_rounds = 5)
r2(train$y, predict(fit_xgb, train$X)) # 0.9927861
r2(test$y, predict(fit_xgb, test$X)) # 0.9912827

partialDiamondsPlot(fit_xgb)


#======================================================================
# lightgbm regression
#======================================================================

dtrain <- lgb.Dataset(train$X, label = train$y)
dtest <- lgb.Dataset(test$X, label = test$y)

params <- list(objective = "regression", 
               metric = "l2",
               learning_rate = 0.01,
               num_leaves = 63,
               min_data_in_leaf = 20,
               bagging_fraction = 0.7,
               bagging_freq = 4)

system.time(fit_lgb <- lgb.train(data = dtrain,
                                 params = params, 
                                 nrounds = 850,
                                 verbose = 0L))
r2(test$y, predict(fit_lgb, test$X)) # 0.991109

partialDiamondsPlot(fit_lgb)


#======================================================================
# ranger regression
#======================================================================

fit_ranger <- ranger(log_price ~ log_carat + cut + color + clarity + depth + table, 
                     data = trainDF, importance = "impurity", num.trees = 500, 
                     always.split.variables = "log_carat", seed = 837363) 
fit_ranger # Estimated R2 0.9887582 

r2(test$y, predict(fit_ranger, testDF)$predictions) # 0.9889626
plot(importance(fit_ranger))
object.size(fit_ranger) # 300 MB

# Effects plots
partialDiamondsPlot(fit_ranger)

