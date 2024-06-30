library(tidyverse)
library(pROC)
source("./settings.R")
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
result = matrix(NA, 22, 8)
colnames(result) = c("SVM", "XGB","RF","DNN", "P", "SEED","VAR", "TYPE")
result <- as.data.frame(result)
result[,"P"] = p
result[,"SEED"] = seeds
result[,"VAR"] = seq(11)
result[, "TYPE"] = c(rep("pval", 11), rep("imp", 11))
model_settings <- list(
  list(generator=generator.lm, type="regression", file_path="./result/LR.base.intp.csv"),
  list(generator=generator.nonlinear, type="regression", file_path="./result/NLR.base.intp.csv"),
  list(generator=generator.glm, type="binary-classification", file_path="./result/GLM.base.intp.csv"),
  list(generator=generator.nglm, type="binary-classification", file_path="./result/NGLM.base.intp.csv")
)
settings <- model_settings[[genmod]]
generator <- settings$generator
type <- settings$type
file_path <- settings$file_path

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
p = c(500, 1000)[p.set]
validate = sample(1:500, 50)
df = generator(seed, p)
x = df$z
y = df$y
trainx = x[-validate,]
trainy = y[-validate]
testx = x[validate,]
testy = y[validate]

# defining the causal set and null set
feature.null <- 11:p
pathway <- list(s = feature.null)
if(type == "binary-classification"){
  train_obj = importDnnet(trainx, as.factor(trainy))
  test_obj = importDnnet(testx, as.factor(testy))
}else{
  train_obj = importDnnet(trainx, trainy)
  test_obj = importDnnet(testx, testy)
}

#### machine learning and permfit ####
i = 1
for(mod_args in args_list){  
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
        shuffle = nshuffle,
        pathway_list = pathway,
        active_var = c(1:10)
      ), 
      mod_args
    )
  )
  result[1:10, i] <- permMod@importance$importance_pval_x[1:10]
  result[11, i] <- permMod@block_importance$importance_pval_x[1]
  result[12:21, i] <- permMod@importance$importance[1:10]
  result[22, i] <- permMod@block_importance$importance[1]
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