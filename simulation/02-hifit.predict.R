library(glmnet)
library(tidyverse)
library(pROC)
source("./settings.R")
source("./HFS.R")
source("./permfit.R")
source("./classes.R")
source("./control.R")

#### Load simulation settings ####
# Please pass three arguments when running the simulation script
# 1: seeds - int, 1-100
# 2: p.set - int, 1-3 (set dim to 500, 1000, 10000)
# 3: genmod - int, 1-4 (LR, NLR, GLM, NGLM)
args <- commandArgs(trailingOnly = TRUE)
seeds <- as.numeric(args[1])
p.set <- as.numeric(args[2])
genmod <- as.numeric(args[3])
p = c(500, 1000, 10000)[p.set]
result = matrix(NA, 10, 7)
colnames(result) = c("MSE","PCC","TV","AV","P","SEED","METHOD")
result <- as.data.frame(result)
result[,"P"] = p
result[,"SEED"] = seeds
result[,"METHOD"] = c("Lasso","S-Lasso","S-XGB","HF-XGB","S-RF","HF-RF","S-SVM","HF-SVM","S-DNN","HF-DNN")
model_settings <- list(
  list(generator=generator.lm, type="regression", file_path="./result/LR.hifit.pred.csv"),
  list(generator=generator.nonlinear, type="regression", file_path="./result/NLR.hifit.pred.csv"),
  list(generator=generator.glm, type="binary-classification", file_path="./result/GLM.hifit.pred.csv"),
  list(generator=generator.nglm, type="binary-classification", file_path="./result/NGLM.hifit.pred.csv")
)
settings <- model_settings[[genmod]]
generator <- settings$generator
type <- settings$type
file_path <- settings$file_path

#### Other settings of the hyperparameters ####
pvacut = 0.1
n_perm = 100
n_ensemble = 100 #100
n_epoch = 1000 #1000
n_tree = 1000
node_size = 3
esCtrl = list(
  n.hidden = c(50, 40, 30, 20), 
  activate = "relu", 
  l1.reg = 10**-4, 
  early.stop.det = 1000, 
  n.batch = 50, 
  n.epoch = n_epoch, 
  learning.rate.adaptive = "adam", 
  plot = FALSE)

args_list = list(
  xgb = list(
    method="xgboost", 
    params=list(
      booster="gbtree", 
      objective=ifelse(type=="regression", "reg:squarederror", "binary:logistic"), 
      eta=0.3, 
      gamma=0, 
      max_depth=5, 
      min_child_weight=1, 
      subsample=1, 
      colsample_bytree=1
    )
  ),
  rf = list(
    method = "random_forest",
    n.ensemble=n_ensemble, 
    ntree=n_tree, 
    nodesize=node_size
  ),
  svm = list(method="svm", n.ensemble=n_ensemble),
  dnn = list(
    method = "ensemble_dnnet",
    n.ensemble = n_ensemble,
    esCtrl = esCtrl,
    verbose = 0
  )
)

#### split training and fitting ####
seed = 6012023 * seeds
set.seed(seed)
validate = sample(1:500, 50)
df = generator(seed, p)
x = df$z
y = df$y
trainx = x[-validate,]
trainy = y[-validate]
testx = x[validate,]
testy = y[validate]

#### running lasso ####
family = ifelse(type=="regression", "gaussian", "binomial")
cv.mod = cv.glmnet(trainx, trainy, alpha = 1, family=family)
best.mod = glmnet(trainx, trainy, alpha = 1, lambda = cv.mod$lambda.min, family=family)
predy = predict(best.mod, testx)
if (type == "regression"){
    result[1, 1] = mean((testy-predy)^2)
    result[1, 2] = cor(testy, predy)
} else{
    y_lab = ifelse(predy > 0.5, 1, 0)
    result[1, 1] = mean(testy == y_lab)
    result[1, 2] = suppressMessages(auc(roc(testy, predy)))
}
result[1, 3] = sum(coef(best.mod)@i %in% c(1:10))
result[1, 4] = sum(coef(best.mod)!=0)

#### feature screening using HFS ####
val.idx <- sample(1:450, 50)
hfs.object <- hybrid.corr(trainx[-val.idx, ], trainy[-val.idx])
idx <- list()
tau.best <- c()
i <- 1
for(mod_args in args_list){
    tau.object <- do.call(
        tune.tau,
        c(list(
            X = trainx,
            y = trainy,
            hfs.object = hfs.object,
            val.idx = val.idx,
            tune.type = "group-permfit"
        ), mod_args)
    )
    idx[[i]] <- tau.object$idx
    tau.best[i] <- tau.object$tau.best
    result$TV[2*i+1] <- sum(tau.object$idx %in% c(1:10))
    result$AV[2*i+1] <- length(tau.object$idx)
    i <- i + 1
}
print(result)

#### fitting the s-lasso ####
trainxn <- trainx[, unique(unlist(idx))]
testxn <- testx[, unique(unlist(idx))]
feature.keep <- unique(unlist(idx))
cv.mod = cv.glmnet(trainxn, trainy, alpha = 1, family=family)
best.mod = glmnet(trainxn, trainy, alpha = 1, lambda = cv.mod$lambda.min, family=family)
predy = predict(best.mod, testxn)
if (type == "regression"){
  result[2, 1] = mean((testy-predy)^2)
  result[2, 2] = cor(testy, predy)
} else{
  y_lab = ifelse(predy > 0.5, 1, 0)
  result[2, 1] = mean(testy == y_lab)
  result[2, 2] = suppressMessages(auc(roc(testy, predy)))
}
result[2, 3] = sum(feature.keep[coef(best.mod)@i] %in% c(1:10))
result[2, 4] = sum(coef(best.mod)!=0)

#### fitting the hfs-machine learning models ####
i = 3
for(mod_args in args_list){
  feature.keep <- idx[[(i-3)/2 + 1]]
  trainxn <- trainx[, feature.keep]
  testxn <- testx[, feature.keep]
  nfeature <- sum(feature.keep %in% seq(10))
  total.feature <- length(feature.keep)
  if(type == "regression"){
    train_obj <- importDnnet(x=trainxn, y=trainy)
    test_obj <- importDnnet(x=testxn, y=testy)
  }else{
    train_obj <- importDnnet(x=trainxn, y=as.factor(trainy))
    test_obj <- importDnnet(x=testxn, y=as.factor(testy))
  }
  
  full_mod = do.call(
    mod_permfit,
    c(list(model.type=type, object=train_obj), mod_args)
  )
  y_pred = predict_mod_permfit(
    mod = full_mod, 
    object = test_obj, 
    method = mod_args$method, 
    model.type = type
  )
  if(type=="regression"){
    result[i, 1:2] = c(mean((y_pred - test_obj@y)^2), 
                       cor(test_obj@y, y_pred, method = "pearson"))
  }else{
    y_lab <- ifelse(y_pred > 0.5, 0, 1)
    result[i, 1:2] = c(
      mean(y_lab == test_obj@y), 
      suppressMessages(auc(roc(testy, y_pred)))
    )
  }
  print(result)
  i = i + 1
  
  # using permfit
  set.seed(seed)
  nshuffle = sample(length(trainy))
  permMod = do.call(
    permfit, 
    c(
      list(
        train = train_obj, 
        k_fold = 5, 
        n_perm = n_perm, 
        shuffle = nshuffle
      ), 
      mod_args
    )
  )
  keyFeature = which(permMod@importance$importance_pval <= pvacut)
  
  if (length(keyFeature) > 1){
    if(type=="regression"){
      final_obj = importDnnet(trainxn[, keyFeature], trainy)
      final_test_obj = importDnnet(testxn[, keyFeature], testy)
    }else{
      final_obj = importDnnet(trainxn[, keyFeature], as.factor(trainy))
      final_test_obj = importDnnet(testxn[, keyFeature], as.factor(testy))
    }
    mod = do.call(
      mod_permfit, 
      c(list(model.type = type, object = final_obj), mod_args)
    )
    y_pred = predict_mod_permfit(
      mod = mod, 
      object = final_test_obj, 
      method = mod_args$method, 
      model.type = type
    )
    if(type=="regression"){
      result[i, 1:4] = c(
        mean((y_pred - test_obj@y)^2), 
        cor(test_obj@y, y_pred, method = "pearson"), 
        sum(feature.keep[keyFeature] %in% c(1:10)),
        sum(length(keyFeature))
      )
    }else{
      y_lab <- ifelse(y_pred > 0.5, 0, 1)
      result[i, 1:4] = c(
        mean(y_lab == test_obj@y), 
        suppressMessages(auc(roc(testy, y_pred))), 
        sum(feature.keep[keyFeature] %in% c(1:10)),
        sum(length(keyFeature))
      )
    }
  }
  else{
    warning("Permfit: No significant feature found!")
  }
  print(result)
  i = i + 1
}

# Check if the file exists
if (file.exists(file_path)) {
  existing_data <- read.csv(file_path)
  combined_data <- rbind(existing_data, as.data.frame(result))
  write.csv(combined_data, file_path, row.names = FALSE)
} else {
  write.csv(as.data.frame(result), file_path, row.names = FALSE)
}
