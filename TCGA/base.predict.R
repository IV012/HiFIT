library(deepTL)
library(glmnet)
library(dplyr)
library(pROC)

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
file_path <- "./result/TCGA.base.pred.csv"

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
validate = sample(dim(df)[1], 0.1 * dim(df)[1])
x = df[, -c(1, 7, 8)]
trainx = x[-validate,]
trainy = y[-validate]
testx = x[validate,]
testy = y[validate]

vars <- apply(trainx, 2, function(x) sd(x))
filter <- vars >= (vars[order(-vars)][1005])
filter[1:5] <- TRUE
filter[is.na(filter)] <- FALSE
trainxn <- trainx[, filter]
testxn <- testx[, filter]

train_obj = importDnnet(trainxn, as.factor(trainy))
test_obj = importDnnet(testxn, as.factor(testy))

#### machine learning ####
i = 1
for(mod_args in args_list){  
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
  result[i, c(1, 2)] = c(mean(y_pred == test_obj@y), 
                  suppressMessages(auc(roc(testy, prob))))
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
