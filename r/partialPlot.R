# Partial dependency plot for xgboost regression object
#'
#' @import xgboost
#' @import graphics
#' 
#' @description Partial dependency plot for xgboost regression object. 
#' Works as long objective is one of "reg:linear", "reg:logistic", "binary:logistic" or "multi:softprob"
#' @author Michael Mayer, \email{mayer@consultag.ch}
#' @param obj xgboost regression model
#' @param pred.data Matrix to be used in prediction (no xgb.DMatrix)
#' @param xname Name of column in \code{pred.data} according to that dependency plot is calculated
#' @param n.pt Evaluation grid size (used only if x is not discrete)
#' @param x.discrete If TRUE, the evaluation grid is set to the unique values of x
#' @param subsample Fraction of lines in pred.data to be used in prediction
#' @param which.class Which class if objective is "multi:softprob" (value from 0 to num_class - 1)
#' @param xlab, ylab, main, type, ... Parameters passed to \code{plot}
#' @param rug Should a rug plot be added?
#' @param seed Random seed used if \code{subsample < 1}

#' @return List of prepared objects
partialPlot <- function(obj, pred.data, xname, n.pt = 19, discrete.x = FALSE, subsample = 1, which.class = NULL,
                        xlab = deparse(substitute(xname)), ylab = "", type = if (discrete.x) "p" else "b",
                        main = paste("Partial Dependence on", deparse(substitute(xname))), 
                        rug = TRUE, seed = NULL, ...) {
  stopifnot(inherits(obj, "xgb.Booster"), dim(pred.data) >= 1)
  
  if (subsample < 1) {
    if (!is.null(seed)) {
      set.seed(seed)
    } 
    n <- nrow(pred.data)
    picked <- sample(n, trunc(subsample * n))
    pred.data <- pred.data[picked, , drop = FALSE]
    
  }
  xv <- pred.data[, xname, drop = FALSE]
  
  if (discrete.x) {
    x <- unique(xv)
  } else {
    x <- seq(min(xv), max(xv), length = n.pt)
  }
  y <- numeric(length(x))
  
  for (i in seq_along(x)) {
    pred.data[, xname] <- x[i]
    
    if (!is.null(which.class)) {
      preds <- predict(obj, pred.data, reshape = TRUE)[, which.class + 1, drop = FALSE]  
    } else {
      preds <- predict(obj, pred.data) 
    }
    
    y[i] <- mean(preds)
  }
  
  plot(x, y, xlab = xlab, ylab = ylab, main = main, type = type, ...)
  rug(pmax(pmin(jitter(xv), max(xv)), min(xv)))
}
