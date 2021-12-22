setwd(choose.dir())

data = read.csv("Feature selection-Copy_wet_season.csv", header=T)
data = data[,-1, drop = FALSE]

#Shuffle the dataframe
data <- data[sample(nrow(data)),]

library(lattice)
library(ggplot2)
library(caret)

library(class)
library(kernlab)
library(neuralnet)
library(randomForest)
#######################################################################
# Training k-NN model
#######################################################################

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
# https://www.rdocumentation.org/packages/caret/versions/6.0-84/topics/trainControl
# http://dataaspirant.com/2017/01/09/knn-implementation-r-using-caret-package/
#set.seed(333)
knn_fit <- train(Chave ~., data , method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)
knn_fit
# repeat with  different 'methods' of KNN (e.g., 'kknn', etc.)
# Use 'tuneGrid' control to check for some specific values as well
knn_pred <- predict(knn_fit, newdata = data)
plot(data$Chave, knn_pred, col='blue', ylab = "Predicted AGB KNN", xlab = "Estimated AGB")
Values <-data.frame(data$Chave, knn_pred)


test_pred <- predict(knn_fit, newdata = test)
pred_Values <-data.frame(test_pred)
# Save file for later use
write.csv(test_pred, file= "KNN_Resultwet.csv")

########################################################################
# Training SVM model
#######################################################################

svm_Linear <- train(Chave ~., data = data, method = "svmRadialCost",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)       #svmLinear

svm_Linear
# repeat with  different 'methods' of SVM (e.g., 'svmRadial', 'svmRadialSigma', etc.)

svm_pred <- predict(svm_Linear, newdata = data)
plot(data$Chave, svm_pred, col='blue', ylab = "Predicted AGB SVM", xlab = "Estimated AGB")
Values <-data.frame(data$Chave, svm_pred)

svm_pred <- predict(svm_Linear, newdata = test)
pred_Values <-data.frame(svm_pred)
write.csv(svm_pred, file= "SVM_Resultwet.csv")

#######################################################################
# Training ANN model
#######################################################################

ann_r_1 <- train(Chave ~., data = data, method = "monmlp",    #monmlp  #brnn
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

ann_r
# repeat with  different 'methods' of ANN (e.g., 'mpl', 'mlpWeightDecay', etc.)

ann_r_pred <- predict(ann_r, newdata = data)
plot(data$Chave, ann_r_pred, col='blue', ylab = "Predicted AGB ANN", xlab = "Estimated AGB")
Values <-data.frame(data$Chave, ann_r_pred)

ann_pred <- predict(ann_r, newdata = test)
write.csv(ann_pred, file= "ANN_Resultwet.csv")

######################################################################
# Training RF model
#######################################################################

rf_r <- train(Chave ~., data = data, method = "ranger",
               trControl=trctrl,
               preProcess = c("center", "scale"),
               tuneLength = 10)

rf_r
# repeat with  different 'methods' of RF (e.g., 'rf', 'Rborist', etc.)

rf_r_pred <- predict(rf_r, newdata = data)
plot(data$Chave, rf_r_pred, col='blue', ylab = "Predicted AGB RF", xlab = "Estimated AGB")
Values <-data.frame(data$Chave, rf_r_pred)

rf_r_pred <- predict(rf_r, newdata = test)
write.csv(rf_r_pred, file= "RF_Resultwet.csv")

#######################################################################
# Training GAMM model
#######################################################################

library(mgcv)
mod.gam <- gam(Chave ~s(Band_7)+s(Band_8)+s(BaCo_3_2)+s(BaCo_5_2)+
                s(BaCo_5_3)+s(BaCo_5_4)+s(BaCo_8A_2), data = data)
summary(mod.gam)
fits = predict(mod.gam, newdata=data, type='response')
plot(data$Chave, fits, col='blue', ylab = "Predicted AGB KNN", xlab = "Estimated AGB")
Values <-data.frame(data$Chave, fits)

mod.gam.gam <- gam(Chave~s(Band_3) + s(Band_7) + s(Band_8) + s(BaCo_3_2) + s(BaCo_5_2) + 
  s(BaCo_5_3) + s(BaCo_5_4) + s(BaCo_8_2) + s(BaCo_8A_2),data = data)

mod.gam.gam <- gam(Chave~ s(Band_7) + s(Band_8) + s(BaCo_3_2) ,data = data)

summary(mod.gam.gam)
fits = predict(mod.gam, newdata=data, type='response')
plot(data$Chave, fits, col='blue', ylab = "Predicted AGB KNN", xlab = "Estimated AGB")

Values_allfunc <-data.frame(data$Chave, fits)
plot(mod.gam)
mod.gam <- gam(Chave~ s(Band_7) + s(Band_8) + s(BaCo_3_2),data = data)
summary(mod.gam)

data_train = data[1:106,]
data_train = data_train[-85:-106,]
data_test = data[85:106,]
model.gam <- gam(Chave~s(Band_3) + s(BaCo_5_3) + s(BaCo_6_2) + s(BaCo_8A_2),data = data)
summary(model.gam)

## Final GAMM model (used in paper)
model.gamm_act <- gamm(Chave~s(Band_3)+ s(Band_8) + s(BaCo_3_2) + s(BaCo_5_3) +
                         s(BaCo_8_2), data = data, correlation=corExp(form = ~ Latitude + Longitude),
                       method = 'REML')
summary(model.gamm_act$gam)
fits = predict(model.gamm_act, newdata=data, type='response')
plot(data$Chave, fits, col='blue', ylab = "Predicted AGB GAMM", xlab = "Estimated AGB")
Values_allfunc <-data.frame(data_test$Chave, fits)

fits = predict(mod.gam, newdata=test, type='response')
write.csv(fits, file= "GAMM_Resultwet.csv")


#################
#Other GAMM model (test)
#################
mod.gam <- gamm(Chave~s(Band_3) + s(Band_7) + s(Band_8) + s(BaCo_3_2) + s(BaCo_5_2) + 
                 s(BaCo_5_3) + s(BaCo_5_4) + s(BaCo_8_2) + s(BaCo_8A_2),data = data)
summary(mod.gam)
fits = predict(mod.gam, newdata=data, type='response')
plot(data$Chave, fits, col='blue', ylab = "Predicted AGB KNN", xlab = "Estimated AGB")


#gamm(formula,random=NULL,correlation=NULL,family=gaussian(),
#     data=list(),weights=NULL,subset=NULL,na.action,knots=NULL,
#     control=list(niterEM=0,optimMethod="L-BFGS-B"),
#     niterPQL=20,verbosePQL=TRUE,method="ML",drop.unused.levels=TRUE,...)

model.gamm <- gamm(Chave~s(Band_3) + s(Band_7,bs="re") + s(Band_8) + s(BaCo_3_2) + s(BaCo_5_2,bs="re") + 
       s(BaCo_5_3,bs="re") + s(BaCo_5_4,bs="re") + s(BaCo_8_2), data = data, 
       correlation=corSpher(form = ~ Latitude + Longitude))

summary(model.gamm$lme)
summary(model.gamm$gam)
plot(model.gamm$gam,pages=1)
predict.gam(model.gamm$lme,data = data)
plot(data$Chave,predict.gam(model.gamm$gam,data = data))


model.gam <- gamm(Chave~s(Band_3) + s(Band_7) + s(Band_8) + s(BaCo_3_2) + s(BaCo_5_2) + 
                s(BaCo_5_3) + s(BaCo_5_4) + s(BaCo_8_2) + s(BaCo_8A_2), data = data)

resid(model$lme, type = "normalized")
summary(model.gam$lme)
summary(model.gam$gam)
plot(model$gam,pages=1)

model.gam.gam <- gam(Chave~s(Band_3) + s(Band_7,bs="cr") + s(Band_8,bs="cr") + s(BaCo_3_2,bs="cr") + s(BaCo_5_2,bs="cr") + 
                     s(BaCo_5_3,bs="cr") + s(BaCo_5_4,bs="cr") + s(BaCo_8_2,bs="cr") + s(BaCo_8A_2,bs="cr"), data = data, select = T, se =T)
summary(model.gam.gam)
summary(model.gamm$gam)
plot(model.gamm$gam,pages=1)


##################################################
#Linear models
##################################################

library(mgcv)
library(caret)
set.seed(0)
b <- train(Chave~(Band_3) + (Band_7) + (Band_8) + (BaCo_3_2) + (BaCo_5_2) + 
             (BaCo_5_3) + (BaCo_5_4) + (BaCo_8_2) + (BaCo_8A_2), 
           data = data,
           method = "glm",
           trControl = trainControl(method = "repeatedcv", number = 5, repeats = 3)
)


b <- train(Chave~(Band_3) + (Band_8) +(BaCo_5_2) + 
             (BaCo_5_3)  + (BaCo_8_2) , 
           data = data,
           method = "gam",
           trControl = trainControl(method = "repeatedcv", number = 5, repeats = 3)
)
print(b)
summary(b$finalModel)

glm(Chave~(Band_3) + (Band_7) + (Band_8) + (BaCo_3_2) + (BaCo_5_2) + 
      (BaCo_5_3) + (BaCo_5_4) + (BaCo_8_2) + (BaCo_8A_2), 
    data = data)
##################################################
