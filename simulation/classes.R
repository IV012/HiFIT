setClass("dnnetInput",
         slots = list(
           x = "matrix",
           y = "ANY",
           w = "numeric"
         ))


setClass("PermFIT",
         slots = list(
           model = "ANY",
           importance = "data.frame",
           block_importance = "data.frame",
           validation_index = "ANY",
           y_hat = "ANY"
         ))