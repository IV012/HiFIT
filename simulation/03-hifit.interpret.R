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
result = matrix(0, 22, 8)
colnames(result) = c("SVM", "XGB","RF","DNN", "P", "SEED","VAR", "TYPE")
result <- as.data.frame(result)
result[,"P"] = p
result[,"SEED"] = seeds
result[,"VAR"] = seq(11)
result[, "TYPE"] = c(rep("pval", 11), rep("imp", 11))
model_settings <- list(
  list(generator=generator.lm, type="regression", file_path="./result/LR.hifit.intp.csv"),
  list(generator=generator.nonlinear, type="regression", file_path="./result/NLR.hifit.intp.csv"),
  list(generator=generator.glm, type="binary-classification", file_path="./result/GLM.hifit.intp.csv"),
  list(generator=generator.nglm, type="binary-classification", file_path="./result/NGLM.hifit.intp.csv")
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
validate = sample(1:500, 50)
df = generator(seed, p)
x = df$z
y = df$y
trainx = x[-validate,]
trainy = y[-validate]
testx = x[validate,]
testy = y[validate]

#### fitting the hfs-machine learning models ####
i = 1
for(mod_args in args_list){
  p_score <- array(NA, dim = c(n_perm, 450, 1))
  p_score2 <- array(NA, dim = c(n_perm, 450, 10))
  for(j in 1:5){
    #### HFS screening ####
    for(k in 1:2){
      if(k == 1){
        val.1 <- seq(45) + 90 * (j - 1)
        val.2 <- seq(46, 90) + 90 * (j - 1)
      }else{
        val.2 <- seq(45) + 90 * (j - 1)
        val.1 <- seq(46, 90) + 90 * (j - 1)
      }
      hfs.object <- hybrid.corr(trainx[-c(val.1, val.2), ], trainy[-c(val.1, val.2)])
      val.idx <- which(seq(dim(trainx)[1])[-val.2] %in% val.1)
      tau.object <- do.call(
        tune.tau,
        c(list(
          X = trainx[-val.2,],
          y = trainy[-val.2],
          hfs.object = hfs.object,
          val.idx = val.idx,
          tune.type = "group-permfit"
        ), mod_args)
      )
      feature.keep <- tau.object$idx
      print(feature.keep)
      trainxn <- trainx[, feature.keep]
      testxn <- testx[, feature.keep]
      nfeature <- sum(feature.keep %in% seq(10))
      total.feature <- length(feature.keep)
      if(type == "regression"){
        train_obj <- importDnnet(x=trainxn[-val.2,], y=trainy[-val.2])
        test_obj <- importDnnet(x=testxn, y=testy)
        val_obj <- importDnnet(x=trainxn[val.2, ], y=trainy[val.2])
      }else{
        train_obj <- importDnnet(x=trainxn[-val.2,], y=as.factor(trainy[-val.2]))
        test_obj <- importDnnet(x=testxn, y=as.factor(testy))
        val_obj <- importDnnet(x=trainxn[val.2, ], y=as.factor(trainy[val.2]))
      }
      
      #### PermFIT Refinement ####
      set.seed(seed)
      nshuffle = sample(length(trainy[-val.2]))
      pathway = ifelse(
        sum(!feature.keep %in% c(1:10)) < 2,
        list(),
        list(null=which(!feature.keep %in% c(1:10))))
      permMod = do.call(
        permfit1, 
        c(
          list(
            train = train_obj, 
            validate = val_obj,
            k_fold = 0, 
            n_perm = n_perm, 
            shuffle = nshuffle,
            active_var = which(feature.keep %in% c(1:10)),
            pathway_list = pathway
          ), 
          mod_args
        )
      )
      p_score2[,val.2,feature.keep[feature.keep <= 10]] <- permMod[[2]]
      if(length(pathway) > 0){
        p_score[,val.2,] <- permMod[[1]]
      }
    }
  }
  #### Using 5-fold Results to Compute P-values and Importance ####
  result[12:21, i] <- apply(apply(p_score2, 2:3, mean, na.rm = TRUE), 2, mean, na.rm = TRUE)
  result[22, i] <- apply(apply(p_score, 2:3, mean, na.rm = TRUE), 2, mean, na.rm = TRUE)
  imp_sd_x <- apply(apply(p_score2, c(1, 3), mean, na.rm = TRUE), 2, stats::sd, na.rm = TRUE)
  imp_sd_block <- apply(apply(p_score, c(1, 3), mean, na.rm=TRUE), 2, stats::sd, na.rm = TRUE)
  result[1:10, i] <- 1 - pnorm(result[12:21, i]/imp_sd_x)
  result[11, i] <- 1 - pnorm(result[22, i]/imp_sd_block)
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
