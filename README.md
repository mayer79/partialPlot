# partialPlot
Partial dependency plots for R objects of type XGBoost, lightGBM and ranger

## Idea
The R function `partialPlot` is used to visualize partial dependency of the response on a covariable. It is inspired by the analogous function in the `randomForest` package and works as long as `predict` returns numeric values (no classes!).

The main arguments of `partialPlot` are as follows
1. `obj`: model object of type `lgb.Booster`, `xgb.Booster` or `ranger`
2. `pred.data`: Matrix to be used in prediction (no special objects like `xgb.DMatrix` or `lgb.Dataset`)
3. `xname`: Name of column in `pred.data` according to that dependency plot is calculated
4. `n.pt`: Evaluation grid size (used only if `x` is not discrete). Quantile cuts are used.
5. `x.discrete`: If TRUE, the evaluation grid is set to the unique values of `x`
6. `subsample`: Fraction of lines in `pred.data` to be used in prediction
7. `which.class`: Which class if objective is "multi:softprob" (value from 0 to num_class - 1)

## The function
Check R/partialPlot.R for parameters etc.

## Examples
### Example 1: Regression (realistic example based on diamonds data set)

```
library(ggplot2) # for data set "diamonds"
library(xgboost)
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


![Diamonds plot](/pics/diamonds.jpeg)


### Example 2: Multiclass prediction (toy example based on iris data set)

```
train <- list(y = as.numeric(iris[, 5]),
              X = as.matrix(iris[, 1:4]))

dtrain <- xgb.DMatrix(train$X, label = as.numeric(train$y) - 1)

param <- list(max_depth = 2, learning_rate = 0.1, objective = "multi:softprob", 
              num_class = 3, eval_metric = "merror")

fit_xgb <- xgb.train(dtrain, params = param, nrounds = 100)

par(mfrow = c(2, 2))

for (nam in colnames(train$X)) {
  partialPlot(fit_xgb, train$X, xname = nam, xlab = "", which.class = 0)
}

```
The effects on species "setosa" (first class, corresponding to level 0) are as follows:
![iris plot](/pics/iris.jpeg)
