############################################
sensitivityAnalysis <- function(model,
                                target = NULL,
                                predictors = NULL, #Data Frame
                                predictorsMeans = NULL, #Data Frame
                                samplesN = 101,
                                from = 0.6,
                                to = 1.4,
                                targetPrediction = "ratio", # c("ratio", "absolute")
                                predictionType = "prediction",
                                level = 0.9) {
    
    if (missing(target)) {
        predictors <- model$model[-1]
        target <- model$model[1]
    }
    targetName <- names(target)
    
    target <- target[[1]]
    
    numberPredictors <- ncol(predictors)     
    
    if (missing(predictorsMeans)) {
        initialPredictorsValues <- predictors %>%
            summarise_all(mean)
        
    } else {
        initialPredictorsValues <- predictorsMeans    
    }
    
    initialTargetValue <- mean(predict(model, newdata = initialPredictorsValues))
    
    sensitivityData <- sapply(initialPredictorsValues, function(x) rep(x, samplesN * numberPredictors))
    changeDF <- seq(from, to, length.out = samplesN)
    
    for (i in seq(1, numberPredictors)) {
        sensitivityData[seq((i-1)*samplesN + 1, i * samplesN), i] <- changeDF * initialPredictorsValues[[i]]    
    }
    sensitivityData <- data.frame(sensitivityData)
    
    predictedTarget <- predict(model,
                               newdata = sensitivityData,
                               interval = predictionType,
                               level = level)
    
    predictedTarget <- as.data.frame(predictedTarget)
    
    if (targetPrediction == "ratio")
        predictedTarget <- mutate_all(predictedTarget, function(x) {x / initialTargetValue})
    
    df <- data.frame(normalized.predictor.change = numeric(0),
                     predictor = character(0),
                     normalized.target.change.fit = numeric(0),
                     normalized.target.change.lower= numeric(0),
                     normalized.target.change.upper = numeric(0)) 
    
    for (i in seq(1, numberPredictors)) {
        df <- rbind(df, 
                    data.frame(
                        normalized.predictor.change = changeDF,
                        predictor = names(predictors)[i],
                        normalized.target.change.fit = predictedTarget$fit[seq((i-1)*samplesN + 1, i * samplesN)],
                        normalized.target.change.lower = predictedTarget$lwr[seq((i-1)*samplesN + 1, i * samplesN)],
                        normalized.target.change.upper = predictedTarget$upr[seq((i-1)*samplesN + 1, i * samplesN)]))
    }
    if (targetPrediction == "ratio") {
        gg <- ggplot(df, aes(x = normalized.predictor.change, 
                             group = predictor)) +
            geom_ribbon(aes(ymin = normalized.target.change.lower,
                            ymax = normalized.target.change.upper,
                            fill = predictor), alpha = 0.1) + 
            geom_vline(xintercept = 1,  color = "grey80") + 
            geom_hline(yintercept = 1, color = "grey80") +
            geom_abline(slope = 1, linetype = "dashed", color = "black",size = 0.8) + 
            geom_abline(slope = -1, intercept = 2, linetype = "dashed", color = "black",size = 0.8) +
            geom_line(aes(x = normalized.predictor.change,
                          y = normalized.target.change.fit,
                          color = predictor), size = 0.8) +
            ylab(paste("Normalized", targetName, "change")) +
            xlab("Normalized Predictor Change") + 
            theme_few()
    } else {
        gg <- ggplot(df, aes(x = normalized.predictor.change, 
                             group = predictor)) +
            geom_ribbon(aes(ymin = normalized.target.change.lower,
                            ymax = normalized.target.change.upper,
                            fill = predictor), alpha = 0.1) + 
            geom_vline(xintercept = 1,  color = "black",size = 0.8) + 
            geom_hline(yintercept = initialTargetValue, color = "black",size = 0.8) +
            geom_line(aes(x = normalized.predictor.change,
                          y = normalized.target.change.fit,
                          color = predictor), size = 0.8) +
            ylab(targetName) + 
            xlab("Normalized Predictor Change") + 
            theme_few()        
    }
    
    variableMeans <- cbind(data.frame(target = initialTargetValue),
                           data.frame(initialPredictorsValues))
    names(variableMeans) <- c(targetName, names(initialPredictorsValues))
    return(list(ggplot = gg,
                predictionData = sensitivityData,
                resultsData = df,
                variableMeans = variableMeans))
}

#####################################
sensitivityAnalysisCaret <- function(model,
                                target = NULL,
                                predictors = NULL, #Data Frame
                                predictorsMeans = NULL, #Data Frame
                                samplesN = 101,
                                from = 0.6,
                                to = 1.4,
                                targetPrediction = "ratio" # c("ratio", "absolute")
                                ) {
    
    if (missing(target)) {
        predictors <- model$model[-1]
        target <- model$model[1]
    }
    targetName <- names(target)
    
    target <- target[[1]]
    
    numberPredictors <- ncol(predictors)     
    
    if (missing(predictorsMeans)) {
        initialPredictorsValues <- predictors %>%
            summarise_all(mean)
        
    } else {
        initialPredictorsValues <- predictorsMeans    
    }
    
    initialTargetValue <- mean(predict(model, newdata = initialPredictorsValues))
    
    sensitivityData <- sapply(initialPredictorsValues, function(x) rep(x, samplesN * numberPredictors))
    changeDF <- seq(from, to, length.out = samplesN)
    
    for (i in seq(1, numberPredictors)) {
        sensitivityData[seq((i-1)*samplesN + 1, i * samplesN), i] <- changeDF * initialPredictorsValues[[i]]    
    }
    sensitivityData <- data.frame(sensitivityData)
    
    predictedTarget <- predict(model,
                               newdata = sensitivityData)
    

    
    if (targetPrediction == "ratio")
        predictedTarget <- predictedTarget / initialTargetValue
    
    df <- data.frame(normalized.predictor.change = numeric(0),
                     predictor = character(0),
                     normalized.target.change.fit = numeric(0)) 
    
    for (i in seq(1, numberPredictors)) {
        df <- rbind(df, 
                    data.frame(
                        normalized.predictor.change = changeDF,
                        predictor = names(predictors)[i],
                        normalized.target.change.fit = predictedTarget[seq((i-1)*samplesN + 1, i * samplesN)]))
    }
    if (targetPrediction == "ratio") {
        gg <- ggplot(df, aes(x = normalized.predictor.change, 
                             group = predictor)) +
            geom_vline(xintercept = 1,  color = "grey80") + 
            geom_hline(yintercept = 1, color = "grey80") +
            geom_abline(slope = 1, linetype = "dashed", color = "grey80") + 
            geom_abline(slope = -1, intercept = 2, linetype = "dashed", color = "grey80") +
            geom_line(aes(x = normalized.predictor.change,
                          y = normalized.target.change.fit,
                          color = predictor)) +
            ylab(paste("Normalized", targetName, "change")) +
            xlab("Normalized Predictor Change") + 
            theme_few()
    } else {
        gg <- ggplot(df, aes(x = normalized.predictor.change, 
                             group = predictor)) +
            geom_vline(xintercept = 1,  color = "grey80") + 
            geom_hline(yintercept = initialTargetValue, color = "grey80") +
            geom_line(aes(x = normalized.predictor.change,
                          y = normalized.target.change.fit,
                          color = predictor)) +
            ylab(targetName) + 
            xlab("Normalized Predictor Change") + 
            theme_few()        
    }
    
    variableMeans <- cbind(data.frame(target = initialTargetValue),
                           data.frame(initialPredictorsValues))
    names(variableMeans) <- c(targetName, names(initialPredictorsValues))
    return(list(ggplot = gg,
                predictionData = sensitivityData,
                resultsData = df,
                variableMeans = variableMeans))
}
library(dplyr)
library(ggplot2)
library(caret)
library(ggthemes)

setwd(choose.dir())

jumpData <- read.csv("Boruto_Importance - Copy.csv", header = TRUE)

#jumpData <- read.csv("Band_1-10_Sentinel_2_wet - Copy.csv", header = TRUE)
jumpData = jumpData[,-8, drop = FALSE]
#jumpData = jumpData[,-11, drop = FALSE]
#jumpData = jumpData[,-12:-57, drop = FALSE]

jumpData <- read.csv("DRY/Important.csv", header = TRUE)

# Linear model
model <- lm(Chave ~ ., jumpData) # scale & log-log
summary(model)


results <- sensitivityAnalysisCaret(model, level = .90, predictionType = "prediction", targetPrediction = "raw")
plot(results$ggplot)
results$variableMeans

# Interactions Linear model
model <- lm(Chave ~ .,jumpData)
summary(model)

results <- sensitivityAnalysis(model, level = .90, predictionType = "prediction")
plot(results$ggplot)

# Polynomial
target <- jumpData$Chave
predictors <- select(jumpData)
model <- lm(jumpHeight ~ poly(maxPower, 3, raw = TRUE) * poly(slope, 3, raw = TRUE) * poly(bodyWeight, 3, raw = TRUE),
            jumpData)
summary(model)
results <- sensitivityAnalysis(model, target, predictors, level = .9, predictionType = "prediction", targetPrediction = "raw")
plot(results$ggplot)


# Caret
target <- jumpData$Chave
predictors <- select(jumpData, maxPower, bodyWeight, slope, pushOffDistance)
ctrl <- trainControl(method="cv", allowParallel = TRUE, number = 5, repeats = 1, verboseIter = TRUE)
model <- train(y = target, x = predictors, 
               method = "gam",
               preProcess = c("center", "scale"),
               trControl = ctrl)

results <- sensitivityAnalysisCaret(model, target, predictors, targetPrediction = "raw")
plot(results$ggplot + ylab("Jump Height"))

