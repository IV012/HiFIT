importDnnet <- function(x, y, w = rep(1, dim(x)[1])) {
  
  new("dnnetInput", x = as.matrix(x), y = y, w = w)
}

getSplitDnnet <- function(split, n) {
  
  if(is.numeric(split) && length(split) == 1 && split < 1)
    split <- sample(n, floor(n * split))
  
  if(is.numeric(split) && length(split) == 1 && split > 1)
    split <- 1:split
  
  if(is.character(split) && length(split) == 1 && split == "bootstrap")
    split <- sample(n, replace = TRUE)
  
  split
}

splitDnnet <-function(object, split) {
  
  split <- getSplitDnnet(split, dim(object@x)[1])
  
  train <- object
  train@x <- as.matrix(object@x[split, ])
  if(class(object@y)[1] != "matrix") {
    train@y <- object@y[split]
  } else {
    train@y <- object@y[split, ]
  }
  train@w <- object@w[split]
  if(class(object) == "dnnetSurvInput")
    train@e <- object@e[split]
  
  valid <- object
  valid@x <- as.matrix(object@x[-split, ])
  if(class(object@y)[1] != "matrix") {
    valid@y <- object@y[-split]
  } else {
    valid@y <- object@y[-split, ]
  }
  valid@w <- object@w[-split]
  if(class(object) == "dnnetSurvInput")
    valid@e <- object@e[-split]
  
  list(train = train, valid = valid, split = split)
}

mod_permfit <- function(method, model.type, object, ...) {
  
  if(method == "ensemble_dnnet") {
    
    mod <- do.call(deepTL::ensemble_dnnet, appendArg(list(...), "object", object, TRUE))
  } else if (method == "random_forest") {
    mod <- do.call(randomForest::randomForest,
                   appendArg(appendArg(list(...), "x", object@x, TRUE), "y", object@y, TRUE))
  } else if (method == "lasso") {
    
    lasso_family <- ifelse(model.type == "regression", "gaussian",
                           ifelse(model.type == "binary-classification", "binomial", "cox"))
    cv_lasso_mod <- glmnet::cv.glmnet(object@x, object@y, family = lasso_family)
    mod <- glmnet::glmnet(object@x, object@y, family = lasso_family,
                          lambda = cv_lasso_mod$lambda[which.min(cv_lasso_mod$cvm)])
  } else if (method == "linear") {
    
    if(model.type == "regression") {
      mod <- stats::lm(y ~ ., data.frame(x = object@x, y = object@y))
    } else if(model.type == "binary-classification") {
      mod <- stats::glm(y ~ ., family = "binomial", data = data.frame(x = object@x, y = object@y))
    } 
  } else if (method == "svm") {
    
    if(model.type == "regression") {
      mod <- e1071::tune.svm(object@x, object@y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                             tunecontrol = e1071::tune.control(cross = 5))
      mod <- mod$best.model
    } else if(model.type == "binary-classification") {
      mod <- e1071::tune.svm(object@x, object@y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                             tunecontrol = e1071::tune.control(cross = 5))
      mod <- e1071::svm(object@x, object@y, gamma = mod$best.parameters$gamma, cost = mod$best.parameters$cost, probability = TRUE)
    } else {
      return("Not Applicable")
    }
  } else if (method == "dnnet") {
    
    spli_obj <- splitDnnet(object, 0.8)
    mod <- do.call(deepTL::dnnet, appendArg(appendArg(list(...), "train", spli_obj$train, TRUE),
                                            "validate", spli_obj$valid, TRUE))
  } else if (method == "xgboost") {
    
    if(model.type == "regression") {
      arg_xg <- list(...) %>%
        appendArg("data", object@x, TRUE) %>%
        appendArg("label", object@y, TRUE) %>%
        appendArg("verbose", 0, TRUE)
      if(!"nrounds" %in% names(arg_xg))
        arg_xg <- arg_xg %>% appendArg("nrounds", 50, TRUE)
      mod <- do.call(xgboost::xgboost, arg_xg)
    } else if(model.type == "binary-classification") {
      arg_xg <- list(...) %>%
        appendArg("data", object@x, TRUE) %>%
        appendArg("label", (object@y == levels(object@y)[1])*1, TRUE) %>%
        appendArg("verbose", 0, TRUE)
      if(!"nrounds" %in% names(arg_xg))
        arg_xg <- arg_xg %>% appendArg("nrounds", 50, TRUE)
      mod <- do.call(xgboost::xgboost, arg_xg)
    } else {
      return("Not Applicable")
    }
  } else {
    
    return("Not Applicable")
  }
  return(mod)
}

predict_mod_permfit <- function(mod, object, method, model.type) {
  
  if(model.type == "regression") {
    
    if(!method %in% c("dnnet", "ensemble_dnnet", "linear", "lasso")) {
      return(predict(mod, object@x))
    } else if(method %in% c("dnnet", "ensemble_dnnet")){
      return(deepTL::predict(mod, object@x))
    } else if(method == "linear") {
      return(predict(mod, data.frame(x = object@x)))
    } else {
      return(predict(mod, object@x)[, "s0"])
    }
  } else if(model.type == "binary-classification") {
    
    if(method == "dnnet") {
      return(deepTL::predict(mod, object@x)[, mod@label[1]])
    } else if(method == "ensemble_dnnet") {
      return(deepTL::predict(mod, object@x)[, mod@model.list[[1]]@label[1]])
    } else if(method == "random_forest") {
      return(predict(mod, object@x, type = "prob")[, 1])
    } else if(method == "lasso") {
      return(1 - predict(mod, object@x, type = "response")[, "s0"])
    } else if (method == "linear") {
      return(1 - predict(mod, data.frame(x = object@x, y = object@y), type = "response"))
    } else if(method == "svm") {
      return(attr(predict(mod, object@x, decision.values = TRUE, probability = TRUE),
                  "probabilities")[, levels(object@y)[1]])
    } else if(method == "xgboost") {
      return(predict(mod, object@x))
    }
  } else {
    return("Not Applicable")
  }
}

log_lik_diff <- function(model.type, y_hat, y_hat0, object, y_max = 1-10**-10, y_min = 10**-10) {
  
  if(model.type == "regression") {
    return((object@y - y_hat)**2 - (object@y - y_hat0)**2)
  } else if(model.type == "binary-classification") {
    y_hat <- ifelse(y_hat < y_min, y_min, ifelse(y_hat > y_max, y_max, y_hat))
    y_hat0 <- ifelse(y_hat0 < y_min, y_min, ifelse(y_hat0 > y_max, y_max, y_hat0))
    return(-(object@y == levels(object@y)[1])*log(y_hat) - (object@y != levels(object@y)[1])*log(1-y_hat) +
             (object@y == levels(object@y)[1])*log(y_hat0) + (object@y != levels(object@y)[1])*log(1-y_hat0))
  } else {
    return("Not Applicable")
  }
}

permfit <- function(train, validate = NULL, k_fold = 5,
                    n_perm = 100, mod_fun=mod_permfit, 
                    predict_fun=predict_mod_permfit,
                    pathway_list = list(),
                    active_var = NULL,
                    method = c("ensemble_dnnet", "random_forest",
                               "lasso", "linear", "svm", "dnnet",
                               "xgboost")[1],
                    shuffle = NULL,...) {
  n_pathway <- length(pathway_list)
  n <- dim(train@x)[1]
  if (is.null(active_var)){
    p <- dim(train@x)[2]
    active_var <- seq(p)
  }
  else{
    p <- length(active_var)
  }
  if(class(train) == "dnnetInput") {
    if(is.factor(train@y)) {
      model.type <- "binary-classification"
    } else {
      model.type <- "regression"
    }
  } else {
    stop("'train' has to be either a dnnetInput object.")
  }
  
  if(k_fold == 0) {
    
    if(is.null(validate))
      stop("A validation set is required when k = 0. ")
    n_valid <- dim(validate@x)[1]
    
    mod <- mod_fun(method, model.type, train, ...)
    f_hat_x <- predict_fun(mod, validate, method, model.type)
    valid_ind <- list(1:length(validate@y))
    y_pred <- f_hat_x
    
    if(n_pathway >= 1) {
      p_score <- array(NA, dim = c(n_perm, n_valid, n_pathway))
      for(i in 1:n_pathway) {
        x_i <- validate@x
        for(l in 1:n_perm) {
          x_i[, pathway_list[[i]]] <- x_i[, pathway_list[[i]]][sample(n_valid), ]
          pred_i <- predict_fun(mod, importDnnet(x = x_i, y = validate@y), method, model.type)
          p_score[l, , i] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
        }
      }
    }
    
    p_score2 <- array(NA, dim = c(n_perm, n_valid, p))
    j <- 1
    for(i in active_var) {
      x_i <- validate@x
      for(l in 1:n_perm) {
        x_i[, j] <- sample(x_i[, j])
        pred_i <- predict_fun(mod, importDnnet(x = x_i, y = validate@y), method, model.type)
        p_score2[l, , j] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
      }
      j <- j + 1
    }
  } else {
    valid_ind <- list()
    if(is.null(shuffle)) shuffle <- sample(n)
    n_valid <- n
    y_pred <- numeric(length(train@y))
    if(n_pathway >= 1)
      p_score <- array(NA, dim = c(n_perm, n_valid, n_pathway))
    p_score2 <- array(NA, dim = c(n_perm, n_valid, p))
    valid_error <- numeric(k_fold)
    for(k in 1:k_fold) {
      train_spl <- splitDnnet(train, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)])
      valid_ind[[k]] <- shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)]
      
      mod <- mod_fun(method, model.type, train_spl$valid, ...)
      f_hat_x <- predict_fun(mod, train_spl$train, method, model.type)
      valid_error[k] <- sum(log_lik_diff(model.type, f_hat_x, f_hat_x, train_spl$train))
      y_pred[valid_ind[[k]]] <- f_hat_x
      if(k == 1) {
        
        final_model <- mod
      } else if(method == "ensemble_dnnet") {
        
        final_model@model.list <- c(final_model@model.list, mod@model.list)
        final_model@loss <- c(final_model@loss, mod@loss)
        final_model@keep <- c(final_model@keep, mod@keep)
      }
      
      if(n_pathway >= 1) {
        for(i in 1:n_pathway) {
          for(l in 1:n_perm) {
            x_i <- train_spl$train@x
            x_i[, pathway_list[[i]]] <- x_i[, pathway_list[[i]]][sample(dim(x_i)[1]), ]
            pred_i <- predict_fun(mod, importDnnet(x = x_i, y = train_spl$train@y), method, model.type)
            p_score[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], i] <- log_lik_diff(model.type, pred_i, f_hat_x, train_spl$train)
          }
        }
      }
      j <- 1
      for(i in active_var) {
        x_i <- train_spl$train@x
        for(l in 1:n_perm) {
          x_i[, j] <- sample(x_i[, j])
          pred_i <- predict_fun(mod, importDnnet(x = x_i, y = train_spl$train@y), method, model.type)
          p_score2[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], j] <- log_lik_diff(model.type, pred_i, f_hat_x, train_spl$train)
        }
        j <- j + 1
      }
    }
    mod <- final_model
    valid_error <- sum(valid_error)/n_valid
  }
  
  if(is.null(colnames(train@x))) {
    imp <- data.frame(var_name = paste0("V", 1:p))
  } else  {
    imp <- data.frame(var_name = colnames(train@x))
  }
  imp$importance <- apply(apply(p_score2, 2:3, mean), 2, mean, na.rm = TRUE)
  imp$importance_sd <- sqrt(apply(apply(p_score2, 2:3, mean), 2, stats::var, na.rm = TRUE)/n_valid)
  imp$importance_pval <- 1 - stats::pnorm(imp$importance/imp$importance_sd)
  if(n_perm > 1) {
    imp$importance_sd_x <- apply(apply(p_score2, c(1, 3), mean), 2, stats::sd, na.rm = TRUE)
    imp$importance_pval_x <- 1 - stats::pnorm(imp$importance/imp$importance_sd_x)
  }
  
  imp_block <- data.frame()
  if(n_pathway >= 1) {
    
    if(is.null(names(pathway_list))) {
      imp_block <- data.frame(block = paste0("P", 1:n_pathway))
    } else {
      imp_block <- data.frame(block = names(pathway_list))
    }
    imp_block$importance <- apply(apply(p_score, 2:3, mean), 2, mean, na.rm = TRUE)
    imp_block$importance_sd <- sqrt(apply(apply(p_score, 2:3, mean), 2, stats::var, na.rm = TRUE)/n_valid)
    imp_block$importance_pval <- 1 - stats::pnorm(imp_block$importance/imp_block$importance_sd)
    if(n_perm > 1) {
      imp_block$importance_sd_x <- apply(apply(p_score, c(1, 3), mean), 2, stats::sd, na.rm = TRUE)
      imp_block$importance_pval_x <- 1 - stats::pnorm(imp_block$importance/imp_block$importance_sd_x)
    }
  }
  
  return(new("PermFIT", model = mod, importance = imp, block_importance = imp_block,
             validation_index = valid_ind, y_hat = y_pred))
}

permfit1 <- function(train, validate = NULL, k_fold = 5,
                    n_perm = 100, mod_fun=mod_permfit, 
                    predict_fun=predict_mod_permfit,
                    pathway_list = list(),
                    active_var = NULL,
                    method = c("ensemble_dnnet", "random_forest",
                               "lasso", "linear", "svm", "dnnet",
                               "xgboost")[1],
                    shuffle = NULL,...) {
  n_pathway <- length(pathway_list)
  n <- dim(train@x)[1]
  if (is.null(active_var)){
    p <- dim(train@x)[2]
    active_var <- seq(p)
  }
  else{
    p <- length(active_var)
  }
  if(class(train) == "dnnetInput") {
    if(is.factor(train@y)) {
      model.type <- "binary-classification"
    } else {
      model.type <- "regression"
    }
  } else {
    stop("'train' has to be either a dnnetInput object.")
  }
  
  if(k_fold == 0) {
    
    if(is.null(validate))
      stop("A validation set is required when k = 0. ")
    n_valid <- dim(validate@x)[1]
    
    mod <- mod_fun(method, model.type, train, ...)
    f_hat_x <- predict_fun(mod, validate, method, model.type)
    valid_ind <- list(1:length(validate@y))
    y_pred <- f_hat_x
    p_score <- array(NA, dim = c(n_perm, n_valid, 1))
    if(n_pathway >= 1) {
      p_score <- array(NA, dim = c(n_perm, n_valid, n_pathway))
      for(i in 1:n_pathway) {
        x_i <- validate@x
        for(l in 1:n_perm) {
          x_i[, pathway_list[[i]]] <- x_i[, pathway_list[[i]]][sample(n_valid), ]
          pred_i <- predict_fun(mod, importDnnet(x = x_i, y = validate@y), method, model.type)
          p_score[l, , i] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
        }
      }
    }
    
    p_score2 <- array(NA, dim = c(n_perm, n_valid, p))
    j <- 1
    for(i in active_var) {
      x_i <- validate@x
      for(l in 1:n_perm) {
        x_i[, j] <- sample(x_i[, j])
        pred_i <- predict_fun(mod, importDnnet(x = x_i, y = validate@y), method, model.type)
        p_score2[l, , j] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
      }
      j <- j + 1
    }
  } else {
    valid_ind <- list()
    if(is.null(shuffle)) shuffle <- sample(n)
    n_valid <- n
    y_pred <- numeric(length(train@y))
    p_score <- array(NA, dim = c(n_perm, n_valid, 1))
    if(n_pathway >= 1)
      p_score <- array(NA, dim = c(n_perm, n_valid, n_pathway))
    p_score2 <- array(NA, dim = c(n_perm, n_valid, p))
    valid_error <- numeric(k_fold)
    for(k in 1:k_fold) {
      train_spl <- splitDnnet(train, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)])
      valid_ind[[k]] <- shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)]
      
      mod <- mod_fun(method, model.type, train_spl$valid, ...)
      f_hat_x <- predict_fun(mod, train_spl$train, method, model.type)
      valid_error[k] <- sum(log_lik_diff(model.type, f_hat_x, f_hat_x, train_spl$train))
      y_pred[valid_ind[[k]]] <- f_hat_x
      if(k == 1) {
        
        final_model <- mod
      } else if(method == "ensemble_dnnet") {
        
        final_model@model.list <- c(final_model@model.list, mod@model.list)
        final_model@loss <- c(final_model@loss, mod@loss)
        final_model@keep <- c(final_model@keep, mod@keep)
      }
      
      if(n_pathway >= 1) {
        for(i in 1:n_pathway) {
          for(l in 1:n_perm) {
            x_i <- train_spl$train@x
            x_i[, pathway_list[[i]]] <- x_i[, pathway_list[[i]]][sample(dim(x_i)[1]), ]
            pred_i <- predict_fun(mod, importDnnet(x = x_i, y = train_spl$train@y), method, model.type)
            p_score[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], i] <- log_lik_diff(model.type, pred_i, f_hat_x, train_spl$train)
          }
        }
      }
      j <- 1
      for(i in active_var) {
        x_i <- train_spl$train@x
        for(l in 1:n_perm) {
          x_i[, j] <- sample(x_i[, j])
          pred_i <- predict_fun(mod, importDnnet(x = x_i, y = train_spl$train@y), method, model.type)
          p_score2[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], j] <- log_lik_diff(model.type, pred_i, f_hat_x, train_spl$train)
        }
        j <- j + 1
      }
    }
    mod <- final_model
    valid_error <- sum(valid_error)/n_valid
  }
  
  return(list(p_score, p_score2))
}