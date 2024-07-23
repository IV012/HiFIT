library(deepTL)
library(glmnet)
library(dplyr)

#### Load real data settings ####
# Please pass one argument when running this script
# 1: seeds - int, 1-100
args <- commandArgs(trailingOnly = TRUE)
seeds <- as.numeric(args[1])
result = matrix(NA, 4, 4)
colnames(result) = c("MSE","PCC","SEED","METHOD")
result <- as.data.frame(result)
result[,"SEED"] = seeds
result[,"METHOD"] = c("XGB","RF","SVM","DNN")
file_path <- "./result/WLOSS.base.pred.csv"

#### loading the preprocessed training data ####
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

train_obj = importDnnet(trainx, trainy)
test_obj = importDnnet(testx, testy)

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
  dnn = list(
    method = "ensemble_dnnet",
    n.ensemble = n_ensemble,
    esCtrl = esCtrl,
    verbose = 0
  )
)

#### machine learning and permfit ####
i = 1
for(mod_args in args_list){  
  full_mod = do.call(
    mod_permfit,
    c(list(model.type="regression", object=train_obj), mod_args)
  )
  y_pred = predict_mod_permfit(
    mod = full_mod, 
    object = test_obj, 
    method = mod_args$method, 
    model.type = "regression"
  )
  result[i, c(1, 2)] = c(mean((y_pred - test_obj@y)^2), 
                  cor(test_obj@y, y_pred, method = "pearson"))
  
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
