library(deepTL)
library(glmnet)
library(tidyverse)
library(pROC)
source("./settings.R")

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
result = matrix(NA, 8, 7)
colnames(result) = c("MSE","PCC","TV", "AV", "P", "SEED","METHOD")
result <- as.data.frame(result)
result[,"P"] = p
result[,"SEED"] = seeds
result[,"METHOD"] = c("XGB-True","XGB","RF-True","RF","SVM-True","SVM","DNN-True","DNN")
model_settings <- list(
  list(generator=generator.lm, type="regression", file_path="./result/LR.base.pred.csv"),
  list(generator=generator.nonlinear, type="regression", file_path="./result/NLR.base.pred.csv"),
  list(generator=generator.glm, type="binary-classification", file_path="./result/GLM.base.pred.csv"),
  list(generator=generator.nglm, type="binary-classification", file_path="./result/NGLM.base.pred.csv")
)
settings <- model_settings[[genmod]]
generator <- settings$generator
type <- settings$type
file_path <- settings$file_path

#### Other settings of the hyperparameters ####
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

#### Split training and fitting ####
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
if(type == "binary-classification"){
  train_obj = importDnnet(trainx, as.factor(trainy))
  test_obj = importDnnet(testx, as.factor(testy))
}else{
  train_obj = importDnnet(trainx, trainy)
  test_obj = importDnnet(testx, testy)
}


#### Machine learning prediction ####
i = 2
for(mod_args in args_list){  
  true_mod =   full_mod = do.call(
    mod_permfit,
    c(list(model.type=type, object=train_obj), mod_args)
  )
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
  if (type=="regression"){
    result[i, 1:4] = c(
      mean((y_pred - test_obj@y)^2), 
      cor(test_obj@y, y_pred, method = "pearson"), 
      10, 500)
  }else if (type=="binary-classification"){
    y_lab <- ifelse(y_pred > 0.5, 0, 1)
    result[i, 1:4] = c(
      mean(y_lab == test_obj@y),
      suppressMessages(auc(roc(testy, y_pred))), 
      10, 500)
  }
  print(result)
  i = i + 2
}

#### Golden standard comparison ####
i = 1
if(type == "binary-classification"){
  train_obj = importDnnet(trainx[, 1:10], as.factor(trainy))
  test_obj = importDnnet(testx[, 1:10], as.factor(testy))
}else{
  train_obj = importDnnet(trainx[, 1:10], trainy)
  test_obj = importDnnet(testx[, 1:10], testy)
}

for(mod_args in args_list){  
  true_mod =   full_mod = do.call(
    mod_permfit,
    c(list(model.type=type, object=train_obj), mod_args)
  )
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
  if (type=="regression"){
    result[i, 1:4] = c(
      mean((y_pred - test_obj@y)^2), 
      cor(test_obj@y, y_pred, method = "pearson"), 
      10, 10)
  }else  if (type=="binary-classification"){
    y_lab <- ifelse(y_pred > 0.5, 0, 1)
    result[i, 1:4] = c(
      mean(y_lab == test_obj@y),
      suppressMessages(auc(roc(testy, y_pred))), 
      10, 10)
  }
  print(result)
  i = i + 2
}

# Check if the file exists
if (file.exists(file_path)) {
  existing_data <- read.csv(file_path)
  combined_data <- rbind(existing_data, as.data.frame(result))
  write.csv(combined_data, file_path, row.names = FALSE)
} else {
  write.csv(as.data.frame(result), file_path, row.names = FALSE)
}
