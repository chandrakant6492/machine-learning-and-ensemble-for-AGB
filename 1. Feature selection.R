
#install.packages('Boruta')
library(ranger)
library(Boruta)
setwd(choose.dir())

data = read.csv("All_variables_wet-season.csv", header=T)

data = data[,-1, drop = FALSE]



### When plotting only with importance variable
#data = read.csv("Feature selection-Copy_wet_season.csv", header=T)
#data = data[,-8, drop = FALSE]

# Boruta feature selection
boruta_output <- Boruta(Chave ~ ., data=data, doTrace=0)
names(boruta_output)

boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif) 

roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance",ylab = "Variable Importance", cex.lab=1.3)  
axis(2,cex.axis=1.5)
axis(1,cex.axis=1.5)

#VIF feature selection
library(corrplot)

M <- cor(data)
corrplot(M, method="circle",type="upper", tl.col = "black", tl.srt = 9)
         
model <-lm(Chave~.,data)
summary(model)
VIF(lm(Band_8~Band_7,data))
cor(data$BaCo_8A_2,data$BaCo_6_2)

library(mctest)
x<-data[,-10]
x<- x[,-9]
y<-data[,9]

## plot with default threshold of VIF and Eigenvalues with no intercept
mc.plot(x, y)

