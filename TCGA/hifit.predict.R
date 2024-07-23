library(glmnet)
library(dplyr)
library(pROC)
source("./code/HFS.R")
source("./code/permfit.R")
source("./code/classes.R")
source("./code/control.R")

#### Load real data settings ####
# Please pass one argument when running this script
args <- commandArgs(trailingOnly = TRUE)
seeds <- as.numeric(args[1])
result = matrix(NA, 10, 5)
colnames(result) = c("MSE","PCC","AV","SEED","METHOD")
result <- as.data.frame(result)
result[,"SEED"] = seeds
result[,"METHOD"] = c("Lasso","S-Lasso", "S-SVM","HF-SVM", "S-XGB","HF-XGB",
                      "S-RF","HF-RF","S-DNN","HF-DNN")
file_path <- "./result/TCGA.hifit.pred.csv"

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
      objective="binary:logistic", 
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
df <- readRDS("cleaned_dat.RDS")
df <- df[!(df$overall_survival < 365*5 & df$status == 0), ]
y <- ifelse(df$overall_survival >= 365*5, 0, 1)

#### split training and fitting ####
seed = 5042023 * seeds
set.seed(seed)
validate = sample(dim(df)[1], 0.2 * dim(df)[1])
x = df[, -c(1, 7, 8)]
trainx = x[-validate,]
trainy = y[-validate]
testx = x[validate,]
testy = y[validate]

#### running lasso ####
cv.mod = cv.glmnet(as.matrix(trainx), trainy, alpha = 1, family="binomial")
best.mod = glmnet(as.matrix(trainx), trainy, alpha = 1,
                  lambda = cv.mod$lambda.min, family="binomial")
prob <- predict(best.mod, as.matrix(testx), type="response")
predy = ifelse(prob > 0.5, 1, 0)
result[1, 1] = mean(testy == predy)
result[1, 2] = suppressMessages(auc(roc(testy, prob)))
result[1, 3] = sum(coef(best.mod)!=0)

#### feature screening ####
val.idx <- sample(seq(dim(trainx)[1]), 30)
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
      tune.type = "group-permfit",
      isAll = TRUE
    ), mod_args)
  )
  idx[[i]] <- tau.object$idx
  tau.best[i] <- tau.object$tau.best
  i <- i + 1
}

feature.keep <- unique(c(1:5, unique(unlist(idx))))
trainxn <- trainx[, feature.keep]
testxn <- testx[, feature.keep]
cv.mod = cv.glmnet(as.matrix(trainxn), trainy, alpha = 1)
best.mod = glmnet(as.matrix(trainxn), trainy, alpha = 1, lambda = cv.mod$lambda.min)
prob <- predict(best.mod, as.matrix(testxn), type="response")
predy = ifelse(prob > 0.5, 1, 0)
result[2, 1] = mean(testy == predy)
result[2, 2] = suppressMessages(auc(roc(testy, prob)))
result[2, 3] = sum(coef(best.mod)!=0)

#### machine learning and permfit ####
i = 3
for(mod_args in args_list){
  feature.keep <- unique(c(1:5, idx[[(i-3)/2 + 1]]))
  trainxn <- trainx[, feature.keep]
  testxn <- testx[, feature.keep]
  total.feature <- length(feature.keep)
  train_obj <- importDnnet(x=trainxn, y=as.factor(trainy))
  test_obj <- importDnnet(x=testxn, y=as.factor(trainy))
  
  full_mod = do.call(
    mod_permfit,
    c(list(model.type="binary-classification", object=train_obj), mod_args)
  )
  prob = predict_mod_permfit(
    mod = full_mod, 
    object = test_obj, 
    method = mod_args$method, 
    model.type = "binary-classification"
  )
  y_pred <- ifelse(prob > 0.5, 0, 1)
  result[i, 1:3] = c(mean(y_pred == test_obj@y), 
                  suppressMessages(auc(roc(testy, prob))), total.feature)
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
  keyFeature <- which(permMod@importance$importance_pval <= pvacut)
  keyFeature <- union(indices, 1:5)
  
  if (length(keyFeature) > 1){
    final_obj = importDnnet(trainxn[, keyFeature], as.factor(trainy))
    final_test_obj = importDnnet(testxn[, keyFeature], as.factor(testy))
    mod = do.call(
      mod_permfit, 
      c(list(model.type = "binary-classification", object = final_obj), mod_args)
    )
    prob = predict_mod_permfit(
      mod = mod, 
      object = final_test_obj, 
      method = mod_args$method, 
      model.type = "binary-classification"
    )
    y_pred <- ifelse(prob > 0.5, 0, 1)
    result[i, 1:3] = c(
      mean(y_pred == test_obj@y), 
      suppressMessages(auc(roc(testy, prob))), 
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
