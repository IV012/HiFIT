library(KPC)
library(isotree)
library(dHSIC)
library(foreach)
library(doParallel)

hybrid.corr <- function(X, y, tau=0.6, prop=0.95, isAll=FALSE, utilities=list(), M=3){
  n <- dim(X)[1]
  if(n != length(y)){
    stop("The length of y does not match the row number of X")
  }
  p <- dim(X)[2]
  v <- length(utilities)
  type <- ifelse(length(unique(y))==2, "binary-classification", "regression")
  
  # use parallel computing to save time
  print(parallel::detectCores())
  if(parallel::detectCores() > 1){
    cl <- parallel::makeCluster(min(parallel::detectCores()-1, 8))
  }else{
    cl <- parallel::makeCluster(parallel::detectCores())
  }
  doParallel::registerDoParallel(cl)
  # compute the default utilities if not specified
  if(v == 0){
    v = 2
    corrs <- matrix(NA, p, 2)
    corrs[, 1] <- foreach(i=1:p, .combine="c") %dopar% {
      x <- X[, i]
      x_powers <- matrix(nrow=length(y), ncol=M)
      for(i in seq(M)){
        x_powers[, i] <- x ^ i
      }
      if (type == "binary-classification"){
        model <- glm(y ~ x_powers, family = "binomial")
        null_model <- glm(y ~ 1, family = "binomial")
        PseudoR2 <- 1 - (logLik(model) / logLik(null_model))
        return(unname(PseudoR2))
      }else {
        return(summary(lm(y~x_powers))$adj.r.squared)
      }
    }
    corrs[, 2] <- foreach(i=1:p, .combine="c") %dopar% {
      rho <- NaN
      try(rho <- KPC::KPCRKHS(Y=y, Z=X[, i]))
    }
    if(sum(is.na(corrs[, 2])) >= floor(p*0.1)){
      corrs[, 2] <- foreach(i=1:p, .combine="c") %dopar% dHSIC::dhsic(X=X[, i], Y=y)$dHSIC
    }
  }else{
    # otherwise compute the user specified utility functions
    corrs <- matrix(NA, p, v)
    for(i in seq(v)){
      u_fun <- utilities[[i]]
      corrs[, i] <- sapply(1:p, function(i) u_fun(X[, i], y))
    }
  }
  suppressWarnings(parallel::stopCluster(cl))
  
  # compute the feature-wise isolation score
  fit.isotree <- isotree::isolation.forest(corrs, ndim=v, ntrees=100, nthreads=1)
  score <- predict(fit.isotree, corrs)
  cutoff <- apply(corrs, 2, function(x) quantile(x, probs=prop, na.rm=TRUE))
  if (isAll){
    cond <- apply(corrs, 1, function(x) all(x >= cutoff))
  }else{
    cond <- apply(corrs, 1, function(x) TRUE %in% c(x>=cutoff))
  }
  idx <- which(cond == TRUE & score >= tau)
  return(list(idx=idx, corrs=corrs, score=score, type=type))
}

log_lik <- function(
    fit.type=c("regression, binay-classification"), y, y_hat, 
    y_max = 1-10**-10, y_min = 10**-10){
  if (fit.type=="regression") {
    return(mean((y-y_hat)^2))
  }else if (fit.type=="binary-regression"){
    y_hat <- ifelse(y_hat < y_min, y_min, ifelse(y_hat > y_max, y_max, y_hat))
    return(-(y == levels(y)[1])*log(y_hat) - (y != levels(y)[1])*log(1-y_hat))
  }else{
    stop("Prediction Type Not Applicable.")
  }
}

tune.tau <- function(X, y, hfs.object, val.idx=NULL,
                     tau.set=c(0.5, 0.55, 0.6, 0.65, 0.7), test_ratio=0.2,
                     mod_fun=mod_permfit, predict_fun=predict_mod_permfit,
                     risk_fun=log_lik,
                     tune.type=c("cross-fitting","group-permfit")[1], 
                     method=c("ensemble_dnnet","random_forest","svm","xgboost")[1],
                     alpha=0.1, prop=0.95, isAll=FALSE, utilities=list(),...){
  
  if(length(tau.set) <= 1){
    stop("The function 'tune.tau' only accept multiple taus. Please use 'hybridFS' instead.")
  }
  tau.set <- tau.set[order(tau.set)]
  # split the data into training and testing sets
  n <- length(y)
  if (is.null(val.idx)){
    val.idx <- sample(n, n*min(test_ratio, 0.9)) 
  }
  X_t <- X[val.idx, ]
  y_t <- y[val.idx]
  X <- X[-val.idx, ]
  y <- y[-val.idx]
  fit.type <- hfs.object$type
  if (fit.type == "binary-classification"){
    y <- as.factor(y)
    y_t <- as.factor(y_t)
  }
  # condition on the quantile
  score <- hfs.object$score
  cutoff <- apply(hfs.object$corrs, 2, function(x) quantile(x, probs=prop, na.rm=TRUE))
  if (isAll){
    cond <- apply(hfs.object$corrs, 1, function(x) all(x >= cutoff))
  }else{
    cond <- apply(hfs.object$corrs, 1, function(x) TRUE %in% c(x>=cutoff))
  }
  # return the candidate feature set
  if (tune.type == "cross-fitting"){
    # cross-fitting strategy to determine the best cutoff
    i = 1
    risks <- rep(NA, length(tau.set))
    for(tau in tau.set){
      fea_idx <- which(score >= tau & cond == TRUE)
      if(length(fea_idx) >= n/5){
        next
      }
      train <- importDnnet(x=X[, fea_idx], y=y)
      validate <- importDnnet(x=X_t[, fea_idx], y=y_t)
      mod <- mod_fun(method, fit.type, train, ...)
      y_p <- predict_fun(mod, validate, method, fit.type)
      risks[i] <- risk_fun(fit.type, y_t, y_p)
      i <- i + 1
    }
    # select the tau with the smallest prediction risk
    tau.best <- tau.set[which.min(risks)]
    if(length(tau.best) == 0) tau.best <- max(tau.set)
    idx <- which(cond == TRUE & hfs.object$score >= tau.best)
  } else if (tune.type == "group-permfit"){
    # fast group permfit to roughly determine the cutoff
    min.tau <- max(tau.set)
    for(tau in tau.set){
      fea_idx <- which(score >= tau & cond == TRUE)
      if(length(fea_idx) <= n/5){
        min.tau <- tau
        break
      }
    }
    print(min.tau)
    tau.set <- tau.set[tau.set >= min.tau]
    score <- score[fea_idx]
    train <- importDnnet(x=X[, fea_idx], y=y)
    validate <- importDnnet(x=X_t[, fea_idx], y=y_t)
    # create blocks of indexes within the intervals of tau.set
    pathway <- list()
    for(i in seq(length(tau.set)-1)){
      block.name <- paste("Block", tau.set[i], sep="")
      block.idx <- which(score >= tau.set[i] & score < tau.set[i+1])
      if (length(block.idx) > 1){
        pathway[[block.name]] <- block.idx
      }
    }
    block.imps <- permfit(
      train=train, validate=validate, k_fold=0, active_var=c(), mod_fun=mod_fun, 
      predict_fun=predict_fun, pathway_list=pathway, method=method, ...)@block_importance$importance_pval_x
    print(block.imps)
    block.keys <- which(stats::p.adjust(block.imps, method="hochberg") <= alpha)
    if(length(block.keys) > 0){
      pathway <- pathway[block.keys]
      idx <- c(fea_idx[which(score >= max(tau.set))], fea_idx[unlist(pathway)])
      tau.best <- as.numeric(gsub("Block", "", names(pathway)[1]))
    }else{
      idx <- fea_idx[which(score>=max(tau.set))]
      tau.best <- max(tau.set)
    }
  } else{
    stop("Tuning Type Not Applicable!")
  }
  return(list(idx=idx, tau.best=tau.best, hfs.object=hfs.object, method=method))
}
