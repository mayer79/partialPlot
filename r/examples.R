#==============================================================
# Regression (realistic example based on diamonds data set)
#==============================================================

library(ggplot2) # for data set "diamonds"
library(xgboost)
source("R/partialPlot.R") # or your path


# Prepare data set for modelling
diamonds <- transform(as.data.frame(diamonds),
                      log_price = log(price),
                      log_carat = log(carat),
                      cut = as.numeric(cut),
                      color = as.numeric(color),
                      clarity = as.numeric(clarity))

r2 <- function(y, pred) {
  1 - var(y - pred) / var(y)  
}

set.seed(3928272)
.in <- sample(c(FALSE, TRUE), nrow(diamonds), replace = TRUE, p = c(0.15, 0.85))
x <- c("log_carat", "cut", "color", "clarity", "depth", "table")
x.train <- as.matrix(diamonds[.in, x])
x.test <- as.matrix(diamonds[!.in, x])
y.train <- diamonds[.in, "log_price"]
y.test <- diamonds[!.in, "log_price"]
dtrain <- xgb.DMatrix(x.train, label = y.train)
dtest <- xgb.DMatrix(x.test, label = y.test)
watchlist <- list(train = dtrain, test = dtest)

param <- list(max_depth = 8, learning_rate = 0.01, nthread = 2, lambda = 0.2, 
              objective = "reg:linear", eval_metric = "rmse", subsample = 0.7,
              monotone_constraints = c(1, 1, 0, 1, 0, 0))

fit_gbm <- xgb.train(param, dtrain, watchlist = watchlist, 
                     nrounds = 850, early_stopping_rounds = 5)
r2(y.train, predict(fit_gbm, dtrain)) # 0.9907
r2(y.test, predict(fit_gbm, dtest)) # 0.9901

par(mfrow = 3:2)
partialPlot(fit_gbm, x.train, xname = "log_carat", subsample = 0.01)
partialPlot(fit_gbm, x.train, xname = "cut", discrete.x = TRUE, subsample = 0.01)
partialPlot(fit_gbm, x.train, xname = "color", discrete.x = TRUE, subsample = 0.01)
partialPlot(fit_gbm, x.train, xname = "clarity", discrete.x = TRUE, subsample = 0.01)
partialPlot(fit_gbm, x.train, xname = "depth", subsample = 0.01)
partialPlot(fit_gbm, x.train, xname = "table", subsample = 0.01)


#==============================================================
# Multiclass prediction (toy example based on iris data set)
#==============================================================

library(xgboost)
source("R/partialPlot.R") # or your path

dat <- as.matrix(iris[, 1:4])
dtrain <- xgb.DMatrix(dat, label = as.numeric(iris$Species) - 1)

param <- list(max_depth = 2, learning_rate = 0.1, objective = "multi:softprob", 
              num_class = 3, eval_metric = "merror")

fit_multi <- xgb.train(dtrain, params = param, nrounds = 100)

par(mfrow = c(2, 2))
for (nam in colnames(dat)) {
  partialPlot(fit_multi, dat, xname = nam, xlab = "", main = nam, which.class = 0)
}
