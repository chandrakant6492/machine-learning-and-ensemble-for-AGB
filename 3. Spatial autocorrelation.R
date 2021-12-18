###################################################
#Spatial autocorrelation
###################################################
setwd(choose.dir())
#data = read.csv("Lat and Long 100_2_UTM.csv", header=T)
data1 = read.csv("Lat and Long 100_2.csv", header=T)
#data = read.csv("Lat and Long 100_2_UTM_1.csv", header=T)
library(nlme)
library(ape)
library(MuMIn)
model <- gls( AGB ~ 1 , data = data1 )
semivario <- Variogram(model, form = ~ Latitude + Longitude, resType = "pearson", metric = 'euclidean', nint = 50)
plot(semivario, smooth = T, xlab="Distance (Lat/Lon degree)")
