---
title: "Practical Machine Learning Assignment"
author: "Zane Lim"
date: "23 March 2015"
output: pdf_document
---

### Executive Summary
In this assignment, I used the Weight Lifting Exercise Dataset to train a predictive model that is able to predict (classify) the classes of a specific weight lifting exercise- Unilateral Dumbbell Biceps Curl. There are five classes- exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3V9lavv8N

I performed preprocessing of predictors, such as Box-Cox, centering and scaling transformations. I also reduced the dimensionality to 52 Principal Components through PCA. I trained a random forest model using 10-fold cross validation (cv) and achieved a cv accuracy and kappa statistic of 97%. Hence, predicting on the test model should give me an error or two at most (out of 20).  

### Preprocessing
First, I load the caret library and datasets. There are two types of missing values- 'NA' and '#DIV/0!'.

```r
library(caret)
train0 <- read.csv("data/pml-training.csv",na.string=c("#DIV/0!","NA"))
test0 <- read.csv("data/pml-testing.csv",na.string=c("#DIV/0!","NA"))
```
There are a total of 160 predictors and 19622 observations in the training set. The response variable is 'classe' and the first 7 variables are for logging/operational purpose. Hence, I would remove them from the predictors.

```r
dim(train0)
```

```
## [1] 19622   160
```

```r
colnames(train0)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```
Removing the first 7 variables.

```r
log_col <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2",
             "cvtd_timestamp","new_window","num_window")
train1 <- train0[,!colnames(train0) %in% log_col]
rownames(train1) <- train0$X
test1 <- test0[,!colnames(test0) %in% log_col]
rownames(test1) <- test0$X
```
Exploring the distribution of classe.

```r
table(train0$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
A first instinct looking at the dataset is that it might have a sparse structure, i.e. lots of missing values or 0, resulting in some predictors not having predictive power due to zero (or near zero) variance. We removed them from the training set.

```r
nzv_df <- nearZeroVar(train1,saveMetrics = T)
nzv_df[nzv_df$nzv==T, ]
```

```
##                        freqRatio percentUnique zeroVar  nzv
## kurtosis_yaw_belt        0.00000    0.00000000    TRUE TRUE
## skewness_yaw_belt        0.00000    0.00000000    TRUE TRUE
## amplitude_yaw_belt       0.00000    0.00509632    TRUE TRUE
## avg_roll_arm            77.00000    1.68178575   FALSE TRUE
## stddev_roll_arm         77.00000    1.68178575   FALSE TRUE
## var_roll_arm            77.00000    1.68178575   FALSE TRUE
## avg_pitch_arm           77.00000    1.68178575   FALSE TRUE
## stddev_pitch_arm        77.00000    1.68178575   FALSE TRUE
## var_pitch_arm           77.00000    1.68178575   FALSE TRUE
## avg_yaw_arm             77.00000    1.68178575   FALSE TRUE
## stddev_yaw_arm          80.00000    1.66649679   FALSE TRUE
## var_yaw_arm             80.00000    1.66649679   FALSE TRUE
## max_roll_arm            25.66667    1.47793293   FALSE TRUE
## min_roll_arm            19.25000    1.41677709   FALSE TRUE
## min_pitch_arm           19.25000    1.47793293   FALSE TRUE
## amplitude_roll_arm      25.66667    1.55947406   FALSE TRUE
## amplitude_pitch_arm     20.00000    1.49831821   FALSE TRUE
## kurtosis_yaw_dumbbell    0.00000    0.00000000    TRUE TRUE
## skewness_yaw_dumbbell    0.00000    0.00000000    TRUE TRUE
## amplitude_yaw_dumbbell   0.00000    0.00509632    TRUE TRUE
## kurtosis_yaw_forearm     0.00000    0.00000000    TRUE TRUE
## skewness_yaw_forearm     0.00000    0.00000000    TRUE TRUE
## max_roll_forearm        27.66667    1.38110284   FALSE TRUE
## min_roll_forearm        27.66667    1.37091020   FALSE TRUE
## amplitude_roll_forearm  20.75000    1.49322189   FALSE TRUE
## amplitude_yaw_forearm    0.00000    0.00509632    TRUE TRUE
## avg_roll_forearm        27.66667    1.64101519   FALSE TRUE
## stddev_roll_forearm     87.00000    1.63082255   FALSE TRUE
## var_roll_forearm        87.00000    1.63082255   FALSE TRUE
## avg_pitch_forearm       83.00000    1.65120783   FALSE TRUE
## stddev_pitch_forearm    41.50000    1.64611151   FALSE TRUE
## var_pitch_forearm       83.00000    1.65120783   FALSE TRUE
## avg_yaw_forearm         83.00000    1.65120783   FALSE TRUE
## stddev_yaw_forearm      85.00000    1.64101519   FALSE TRUE
## var_yaw_forearm         85.00000    1.64101519   FALSE TRUE
```

```r
nzv_col <- which(nzv_df$nzv==T)
train1 <- train1[,-nzv_col]
test1 <- test1[,-nzv_col]
test1 <- data.frame(sapply(test1,function(col) as.numeric(col)))
```
Further cleaning the dataset by changing NA values to 0.

```r
train1[is.na(train1)] <- 0
test1[is.na(test1)] <- 0
train1_pred <- train1[,-ncol(train1)]
train1_res <- train1[,ncol(train1)]
test1_pred <- test1[,-ncol(test1)]
```
I then performed Box-cox, centering and scaling transformations, as well as PCA to further "compress" the dataset.

```r
prePro <- preProcess(train1_pred,method=c("BoxCox","center","scale","pca"))
prePro
```

```
## 
## Call:
## preProcess.default(x = train1_pred, method = c("BoxCox",
##  "center", "scale", "pca"))
## 
## Created from 19622 samples and 117 variables
## Pre-processing: Box-Cox transformation, centered, scaled,
##  principal component signal extraction 
## 
## Lambda estimates for Box-Cox transformation:
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##   0.900   1.175   1.450   1.450   1.725   2.000     115 
## 
## PCA needed 52 components to capture 95 percent of the variance
```

```r
train1_pred <- predict(prePro,train1_pred)
test1_pred <- predict(prePro,test1_pred)
```
As shown, the number of predictors are compressed from 118 to 52 principal components (PC) which is quite substantial.

### Modeling
I used 10-fold cross validations with 10 repetitions to train and tune the model.

```r
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)
```
I trained a random forest model on the training set and using parallel computation to speed up the training process. 

```r
library(doMC)
help(package="doMC")
registerDoMC(cores = 4)
fit1 <- train(classe~.,data=cbind(train1_pred,classe=train1_res),method="rf",
              trControl=fitControl,verbose=T)
```

```r
pred1 <- predict(fit1,newdata=test1_pred)
fit1
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## 
## Summary of sample sizes: 17660, 17660, 17659, 17659, 17660, 17660, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9741155  0.9672459  0.003625660  0.004589429
##   27    0.9718985  0.9644451  0.003871334  0.004899456
##   52    0.9619559  0.9518708  0.004566898  0.005775271
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
pred1
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
The tuning parameter in this case is mtry, the number of variables randomly sampled as candidate at each split in trees. From the cv accuracy, surprisingly the optimal mtry is 2, which means our PCs are fairly effective as the top 2 PCs have so much predictive power. The mean CV accuracy and kappa are about 97%. I expect one error during the evaluation of test set (out of 20).

