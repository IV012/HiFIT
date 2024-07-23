library(glmnet)
library(dplyr)
source("./code/HFS.R")
source("./code/permfit.R")
source("./code/classes.R")
source("./code/control.R")

#### Load real data settings ####
# Please pass one argument when running this script
args <- commandArgs(trailingOnly = TRUE)
seeds <- as.numeric(args[1])
result = matrix(NA, 10, 5)
colnames(result) = c("MSE","PCC", "AV", "SEED","METHOD")
result <- as.data.frame(result)
result[,"SEED"] = seeds
result[,"METHOD"] = c("Lasso","S-Lasso", "S-SVM","HF-SVM","S-XGB","HF-XGB",
                      "S-RF","HF-RF","S-DNN","HF-DNN")
file_path <- "./result/WLOSS.hifit.pred.csv"
type = "regression"

#### model settings ####
pvacut = 0.1
n_ensemble = 100 #100
n_perm = 100 #100
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
  svm = list(method="svm", n.ensemble=n_ensemble),
  xgb = list(
    method="xgboost", 
    params=list(
      booster="gbtree", 
      objective="reg:squarederror", 
      eta=0.3, 
      gamma=0, 
      max_depth=6, 
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
  dnn = list(
    method = "ensemble_dnnet",
    n.ensemble = n_ensemble,
    esCtrl = esCtrl,
    verbose = 0
  )
)

#### data generators ####
df <- readRDS("WLOSS.RDS")

#### split training and fitting ####
seed = 5042023 * seeds
set.seed(seed)
validate = sample(dim(df)[1], 0.1*dim(df)[1])
x = df %>% select(-y)
y = df$y %>% unlist()
trainx = x[-validate,]
trainy = y[-validate]
testx = x[validate,]
testy = y[validate]

#### running lasso ####
cv.mod = cv.glmnet(as.matrix(trainx), trainy, alpha = 1)
best.mod = glmnet(as.matrix(trainx), trainy, alpha = 1, lambda = cv.mod$lambda.min)
# best.mod = glmnet(as.matrix(trainx), trainy, alpha = 1)
# which(coef(best.mod)!=0, arr.ind=T)
predy = predict(best.mod, as.matrix(testx))
result[1, 1] = mean((testy-predy)^2)
result[1, 2] = cor(testy, predy)
result[1, 3] = sum(coef(best.mod)!=0)

#### feature screening ####
val.idx <- sample(seq(dim(trainx)[1]), 10)
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
  i <- i + 1
}

#### fitting s-lasso ####
feature.keep <- unique(c(1:9, unique(unlist(idx))))
trainxn <- trainx[, feature.keep]
testxn <- testx[, feature.keep]
cv.mod = cv.glmnet(as.matrix(trainxn), trainy, alpha = 1)
best.mod = glmnet(as.matrix(trainxn), trainy, alpha = 1, lambda = cv.mod$lambda.min)
predy = predict(best.mod, as.matrix(testxn))
result[2, 1] = mean((testy-predy)^2)
result[2, 2] = cor(testy, predy)
result[2, 3] = sum(coef(best.mod)!=0)

#### fitting the hfs-machine learning models ####
i = 3
for(mod_args in args_list){
  feature.keep <- unique(c(1:9, idx[[(i-3)/2 + 1]]))
  trainxn <- trainx[, feature.keep]
  testxn <- testx[, feature.keep]
  total.feature <- length(feature.keep)
  train_obj <- importDnnet(x=trainxn, y=trainy)
  test_obj <- importDnnet(x=testxn, y=testy)

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
  result[i, 1:3] = c(mean((y_pred - test_obj@y)^2), 
                     cor(test_obj@y, y_pred, method = "pearson"), 
                     total.feature)
  print(result)
  i = i + 1
  
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
  keyFeature = unique(c(1:9, keyFeature))
  
  if (length(keyFeature) > 1){
    final_obj = importDnnet(trainxn[, keyFeature], trainy)
    final_test_obj = importDnnet(testxn[, keyFeature], testy)
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
    result[i, 1:3] = c(
      mean((y_pred - test_obj@y)^2), 
      cor(test_obj@y, y_pred, method = "pearson"), 
      sum(length(keyFeature))
    )
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
