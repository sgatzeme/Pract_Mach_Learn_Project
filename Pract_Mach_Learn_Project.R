setwd("C:/Users/desgatz1/Documents/03_Development/_Learning/Self_Learning/Data_Science_Specialzation/00_Administration/Working_Directory")

##========================================================================================== ###
### The goal of your project is to predict the manner in which they did the exercise. 
### This is the "classe" variable in the training set. You may use any of the other variables
### to predict with. You should create a report describing how you built your model, how you 
### used cross validation, what you think the expected out of sample error is, and why you 
### made the choices you did. You will also use your prediction model to predict 20 different 
### test cases. 

### 1. Your submission should consist of a link to a Github repo with your R markdown and 
### compiled HTML file describing your analysis. Please constrain the text of the writeup to 
### < 2000 words and the number of figures to be less than 5. It will make it easier for the 
### graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online 
### (and you always want to make it easy on graders :-).
 
### 2. You should also apply your machine learning algorithm to the 20 test cases available
### in the test data above. Please submit your predictions in appropriate format to the 
### programming assignment for automated grading. See the programming assignment for 
### additional details. 

### =================================================================================================================================== ###
 
# Our outcome variable is classe, a factor variable with 5 levels. For this data set, "participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions:
#     - exactly according to the specification (Class A)
# - throwing the elbows to the front (Class B)
# - lifting the dumbbell only halfway (Class C)
# - lowering the dumbbell only halfway (Class D)
# - throwing the hips to the front (Class E)"

#` r citep(bib['devtools']) `
## ----------------------------------------------------- Setting up Base parameter ----------------------------------------------------- ##

# Load frequently used functions
source("./common/misc.R")

# Log session information for debugging
origLocale <- Sys.getlocale("LC_TIME")
session <- sessInfo(clear=TRUE, loc=c("LC_TIME", "English"), log=TRUE)

# Load required libraries
library(ggplot2)
library(caret)
library(randomForest)
library(rpart)
library(e1071)
library(knitr)
library(knitcitations)

# Change R Options
cleanbib()
options("citation_format" = "pandoc")
options(scipen=999)

# Set up current project directory
inputDir <- file.path(getwd(), "Pract_Mach_Learn")

# Set up data directory
dataDir <- file.path(inputDir,"data")

citep("10.1007/978-3-642-34459-6_6")

## -------------------------------------------------------- Preprocess data ------------------------------------------------------------ ##

# Download source file
downloadFile(dataDir, "training_set.csv", 
             "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

downloadFile(dataDir, "testing_set.csv", 
             "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

trainFile <- file.path(dataDir, "training_set.csv")
    
testFile <- file.path(dataDir, "testing_set.csv")

# Read in activity data
data.train <- read.csv(trainFile, header = TRUE, sep = ",", na.strings = getNaStrings(), stringsAsFactors = FALSE)
data.test <- read.csv(testFile, header = TRUE, sep = ",", na.strings = getNaStrings(), stringsAsFactors = FALSE)

# guaranteed reproducibility
set.seed(230215)

data.train.nearzero <- nearZeroVar(data.train, saveMetrics = TRUE)
data.train <- data.train[, !data.train.nearzero$nzv]
data.train <- data.train[!duplicated(data.train), -c(1:6)]
data.train <- data.train[, colSums(is.na(data.train)) == 0]

data.test.nearzero <- nearZeroVar(data.test, saveMetrics = TRUE)
data.test <- data.test[, !data.test.nearzero$nzv]
data.test <- data.test[!duplicated(data.test), -c(1:6)]
data.test <- data.test[, colSums(is.na(data.test)) == 0]
data.test <- data.test[, !(names(data.test) %in% c("problem_id"))]

corr.train <- cor(na.omit(data.train[sapply(data.train, is.numeric)]))
corr.remove <- findCorrelation(corr.train, cutoff = .90, verbose = FALSE)

data.train <- data.train[, -corr.remove]
data.test <- data.test[, colnames(data.test) %in% colnames(data.train)]

split <- createDataPartition(y=data.train$classe, p=0.8, list=FALSE) 
data.train.train <- data.train[split, ] 
data.train.test <- data.train[-split, ]

qplot(classe, data=data.train, geom="histogram")

md_rf <- randomForest(as.factor(classe) ~ ., data=data.train.train, method="class")
pred_rf <- predict(md_rf, data.train.test, type = "class")
con_rf <- confusionMatrix(pred_rf, data.train.test$classe)

md_rp <- rpart(as.factor(classe) ~ ., data=data.train.train, method="class")
pred_rp <- predict(md_rp, data.train.test, type = "class")
con_rp <- confusionMatrix(pred_rp, data.train.test$classe)

md_nb <- naiveBayes(as.factor(classe) ~ ., data=data.train.train)
pred_nb <- predict(md_nb, data.train.test, type = "class")
con_nb <- confusionMatrix(pred_nb, data.train.test$classe)

md_sv <- svm(as.factor(classe) ~ ., data=data.train.train, gamma = 0.1)
pred_sv <- predict(md_sv, data.train.test, type = "class")
con_sv <- confusionMatrix(pred_sv, data.train.test$classe)

pred_results <- data.frame(random_forest=pred_rf,
                           recursive_partitioning=pred_rp,
                           naive_bayes=pred_nb,
                           support_vector_machine=pred_sv                           
                           )

acc_results <- data.frame(model=c("random_forest", "recursive_partitioning", "naive_bayes", "support_vector_machines"),
                          accuracy=c(con_rf$overall[1], con_rp$overall[1], con_nb$overall[1], con_sv$overall[1]))

acc_results <- acc_results[order(-acc_results$accuracy),]

final_rf <- predict(md_rf, data.test, type="class")
final_sv <- predict(md_sv, data.test, type="class")

final_results <- data.frame(problem_id=c(1:20),
                            random_forest=final_rf,
                            support_vector_machine=final_sv,
                            check=ifelse(final_rf == final_sv,TRUE,FALSE)
                            )

writeOutputfiles = function(x){
    n = length(x)
    for(i in 1:n){
        filename = file.path(dataDir, paste0("case_",i,".txt"))   
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

writeOutputfiles(final_results$random_forest)

bibliography("html")

## include word count - in chunks?

# Set back to original locale
session <- sessInfo(clear=TRUE, loc=c("LC_TIME",origLocale), log=TRUE)