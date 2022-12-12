library(haven)
library(purrr)
library(REEMtree)
library(tree)
library(randomForest)
library(dplyr)
library(plm)
library(stargazer)
library(gbm)
library(plotmo)
library(caret)

# Set Directory

rm(list=ls())

setwd('/Users/paulmcdermott/My Drive/Master 2 Courses/Machine learning for economics /PISA/Data')
#setwd("~/Documents/TSE/Courses/M2/Semester 1/Machine Learning/Project")


# Read in data, clean column names, remove missing data
student_level_df <- read_sas('student_PISA2015_main.sas7bdat')
school_level_df <- read_sas('school_PISA2015_main.sas7bdat')


keep_columns_student <- c('CNTSCHID','CNT','ST004D01T','ESCS','IMMIG','ST071Q02NA','HISCED',
                          'ST076Q06NA','ST076Q11NA','DISCLISCI','TEACHSUP','MMINS','BELONG','MOTIVAT','ANXTEST','COOPERATE','EMOSUPS',
                          'CULTPOSS','HEDRES','PV3MATH')

keep_columns_school <- c('SC064Q02TA', 'SC002Q01TA', 'SC002Q02TA', 'SC004Q02TA', 'SC012Q01TA',
                         'SC016Q01TA', 'SC017Q02NA', 'SC017Q06NA', 'SC018Q01TA01', 'SC048Q02NA',
                         'SC048Q03NA', 'SC061Q01TA', 'SC061Q03TA', 'SC061Q07TA', 'SC064Q03TA', 
                         'SC013Q01TA', 'SC017Q08NA', 'CNT', 'CNTSCHID', 'SC009Q01TA', 'SC009Q13TA')


fr_students_df <- subset(student_level_df,CNT == 'FRA',select = keep_columns_student)
fr_school_df <- subset(school_level_df,CNT == 'FRA',select = keep_columns_school)

write.csv(fr_students_df, file = "students_france.csv")
students_df <- read.csv("students_france.csv", header=T)

students_df <- na.omit(students_df)
students_df <- students_df[,-1]


students_df <- students_df %>%
  rename("SchoolID" = "CNTSCHID",
         "Gender" = "ST004D01T",
         "MathHW"="ST071Q02NA",
         "ParentalEduc"="HISCED",
         "VideoGames"="ST076Q06NA",
         "Sports"="ST076Q11NA",
         "DiscClimate"="DISCLISCI",
         "math_PV" = "PV3MATH")

# Normalise dependent variable
students_df$math_PV <- scale(students_df$math_PV)

###### School Cleaning

write.csv(fr_school_df, file = "school_france.csv")
school_df <- read.csv("school_france.csv", header=T)

# we omit level 5 info, lots of missing values

school_df <- school_df %>%
  rename("ParentParticip" = "SC064Q02TA", 
         "EnrollmentBoys" = "SC002Q01TA",
         "EnrollmentGirls" = "SC002Q02TA",
         "NumComputers" = "SC004Q02TA",
         "AdmissionPerformance" =  "SC012Q01TA",
         "PercGovtFunds" = "SC016Q01TA",
         "InadeqTeachers" = "SC017Q02NA",
         "InadeqMaterials" = "SC017Q06NA",
         "TotalTeachersFT" = "SC018Q01TA01",
         "SpecialNeedsNum" = "SC048Q02NA",
         "DisadvStudsNum" = "SC048Q03NA",
         "Truancy" = "SC061Q01TA",
         "StudentDisrepect" = "SC061Q03TA",
         "TeacherAbsenteeism" = "SC061Q07TA",
         "ParentParticipGovt" = "SC064Q03TA",
         "PublicPrivate" =  "SC013Q01TA",
         "InfraIssues" =  "SC017Q08NA",
         "SchoolID" = "CNTSCHID",
         "GoalsResults" = "SC009Q01TA",
         "DiscussGoals" = "SC009Q13TA")

school_df <- school_df[,-1]

# data transformations
# total students
school_df$NmbStudents <- school_df$EnrollmentBoys + school_df$EnrollmentGirls
school_df$CompsPerStudent <- school_df$NumComputers/school_df$NmbStudents
school_df$StudentTeacherRatio <- school_df$NmbStudents/school_df$TotalTeachersFT

colnames(school_df)

school_df <- select(school_df, -c("EnrollmentBoys", "EnrollmentGirls", "NumComputers",
                                  "TotalTeachersFT"))

school_df <- na.omit(school_df)

#######################
# REEM Tree Replication
#######################
set.seed(22234110)
treeREEM <- REEMtree(math_PV ~ Gender + ESCS + factor(IMMIG) + MathHW + 
                       ParentalEduc + VideoGames + Sports + DiscClimate + 
                       TEACHSUP + MMINS + BELONG + MOTIVAT + ANXTEST + 
                       COOPERATE + EMOSUPS + CULTPOSS + HEDRES,
                  data = students_df, random = ~1|SchoolID)

plot(treeREEM)
text(treeREEM, pretty = 0)


treeREints <- as.data.frame(treeREEM$RandomEffects)
treeREints <- cbind(SchoolID = rownames(treeREints), treeREints)

treeREints <- treeREints %>%
  rename("RandEffectREEM" = "(Intercept)")

treeREints$SchoolID <- as.integer(treeREints$SchoolID)


school_df_ints <- inner_join(school_df, treeREints, by = 'SchoolID')

#######################
# FE Regression
#######################


regFE <- plm(math_PV ~ factor(Gender) + ESCS + factor(IMMIG) + MathHW + 
               ParentalEduc + VideoGames + Sports + DiscClimate + TEACHSUP + 
               MMINS + BELONG + MOTIVAT + ANXTEST + COOPERATE + EMOSUPS + 
               CULTPOSS + HEDRES,
             data = students_df, model = c("within"), index = "SchoolID")

summary(regFE)
stargazer(regFE, type="latex")

regFEints <- as.data.frame(fixef(regFE))
regFEints <- cbind(SchoolID = rownames(regFEints), regFEints)

regFEints <- regFEints %>%
  rename("FixedEffectReg" = "fixef(regFE)")

regFEints$SchoolID <- as.integer(regFEints$SchoolID)

school_df_ints <- inner_join(school_df_ints, regFEints, by = 'SchoolID')

# fixed effects from the two are highly correlated
cor(school_df_ints$FixedEffectReg, school_df_ints$RandEffectREEM)

# clear relationship. Reg FE ones are not centred at 0 as model has no constant
# tree essentially has a constant 
plot(school_df_ints$FixedEffectReg, school_df_ints$RandEffectREEM)

################
# Random Forests
################

# we extend the paper by tuning mtry i.e. the number
# of covariates kept by the random forest at each split (with fixed values for 
# the other hyper-parameters). Method: k-fold cross-validation

# REEM Tree cross-validation
# Set the number of folds for cross-validation
nfolds <- 10

# Set the range of mtry values to test
num_variables = ncol(select(school_df_ints,-c(SchoolID, RandEffectREEM, 
                                              CNT, FixedEffectReg)))
mtry_values <- seq(1,num_variables,1)

# break data into x and y
x_school_df <- select(school_df_ints,-c(SchoolID, RandEffectREEM, CNT, 
                                        FixedEffectReg))
y_school_REEM_df <-school_df_ints$RandEffectREEM

# Perform cross-validation
rf_reem_cv <- train(x = x_school_df, y = y_school_REEM_df,
                 method = "rf",
                 trControl = trainControl(method = "cv", number = nfolds),
                 tuneGrid = expand.grid(mtry = mtry_values))
print(rf_reem_cv$bestTune$mtry)


# use tuned mtry to build "optimal" random forest on full data
REEM_forest_opt <- randomForest(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                                data=school_df_ints,
                                mtry = rf_reem_cv$bestTune$mtry,
                                importance = TRUE)

# FE Tree cross-validation
# y data will be the fixed effect column this time
y_school_FE_df <- school_df_ints$FixedEffectReg

# TODO Perform cross-validation
#rf_fe_cv <- train(x = x_school_df, y = y_school_FE_df,
#                    method = "rf",
#                    trControl = trainControl(method = "cv", number = nfolds),
#                    tuneGrid = expand.grid(mtry = mtry_values))
# why the error, this isn't classification?
#print(rf_fe_cv$bestTune$mtry)

# use tuned mtry to build random forest on full data, this time with fixed
# effects from linear fe model, using optimal mtry from REEM
FE_forest_opt <- randomForest(FixedEffectReg ~ . -SchoolID - RandEffectREEM - CNT, 
                              data=school_df_ints,
                              mtry = rf_reem_cv$bestTune$mtry,
                              importance = TRUE)

# Data for graphs comparing variable importance for each stage 1 effects type

REEM_forest_varimp <- as.data.frame(importance(REEM_forest_opt))
REEM_forest_varimp <- REEM_forest_varimp %>% arrange(desc(IncNodePurity))
REEM_forest_varimp <- head(REEM_forest_varimp,10)
REEM_forest_varimp <- REEM_forest_varimp %>% arrange(IncNodePurity)

FE_forest_varimp <- as.data.frame(importance(FE_forest_opt))
FE_forest_varimp <- FE_forest_varimp %>% arrange(desc(IncNodePurity))
FE_forest_varimp <- head(FE_forest_varimp,10)
FE_forest_varimp <- FE_forest_varimp %>% arrange(IncNodePurity)

# Partial dependence plots
par(mfrow = c(1,2))
partialPlot(REEM_forest_opt, pred.data = school_df_ints, 
            x.var = "DisadvStudsNum", main = "Stage 1: Regression Tree", 
            xlab = "% Disadvantaged Students")
partialPlot(FE_forest_opt, pred.data = school_df_ints, x.var = "DisadvStudsNum", 
            main = "Stage 1: FE Regression", xlab = "% Disadvantaged Students")

################
# Boosting
################

# TODO replace test train with cross-validation

# For the REEM trees, we extend the paper by tuning interaction depth
# with fixed values for the other hyper-parameters. First, we train 7 boosting
# regression trees with with 7 different interaction depths
REEM_boost_d1 = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = train_schools, 
                    distribution = "gaussian", 
                    n.trees = 5000, 
                    interaction.depth = 1,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)
REEM_boost_d2 = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                   data = train_schools, 
                   distribution = "gaussian", 
                   n.trees = 5000, 
                   interaction.depth = 2,
                   shrinkage = 0.001,
                   n.minobsinnode = 3)
REEM_boost_d3 = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = train_schools, 
                    distribution = "gaussian", 
                    n.trees = 5000, 
                    interaction.depth = 3,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)
REEM_boost_d4 = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = train_schools, 
                    distribution = "gaussian", 
                    n.trees = 5000, 
                    interaction.depth = 4,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)
REEM_boost_d5 = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = train_schools, 
                    distribution = "gaussian", 
                    n.trees = 5000, 
                    interaction.depth = 5,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)
REEM_boost_d6 = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = train_schools, 
                    distribution = "gaussian", 
                    n.trees = 5000, 
                    interaction.depth = 6,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)
REEM_boost_d7 = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = train_schools, 
                    distribution = "gaussian", 
                    n.trees = 5000, 
                    interaction.depth = 7,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)

# estimate on test set with each trained tree
boost_estimate_d1 = predict(REEM_boost_d1, 
                         newdata = test_schools, 
                         n.trees = 5000)
boost_estimate_d2 = predict(REEM_boost_d2, 
                            newdata = test_schools, 
                            n.trees = 5000)
boost_estimate_d3 = predict(REEM_boost_d3, 
                            newdata = test_schools, 
                            n.trees = 5000)
boost_estimate_d4 = predict(REEM_boost_d4, 
                            newdata = test_schools, 
                            n.trees = 5000)
boost_estimate_d5 = predict(REEM_boost_d5, 
                            newdata = test_schools, 
                            n.trees = 5000)
boost_estimate_d6 = predict(REEM_boost_d6, 
                            newdata = test_schools, 
                            n.trees = 5000)
boost_estimate_d7 = predict(REEM_boost_d7, 
                            newdata = test_schools, 
                            n.trees = 5000)


# calculate MSE and observe which is minimum (interaction depth = 1)
mean((boost_estimate_d1 - test_schools$RandEffectREEM)^2)
mean((boost_estimate_d2 - test_schools$RandEffectREEM)^2)
mean((boost_estimate_d3 - test_schools$RandEffectREEM)^2)
mean((boost_estimate_d4 - test_schools$RandEffectREEM)^2)
mean((boost_estimate_d5 - test_schools$RandEffectREEM)^2)
mean((boost_estimate_d6 - test_schools$RandEffectREEM)^2)
mean((boost_estimate_d7 - test_schools$RandEffectREEM)^2)

# make boosting regression tree with tuned interaction depth, on full data
REEM_boost_opt = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = school_df_ints, 
                    distribution = "gaussian", 
                    n.trees = 5000, 
                    interaction.depth = 1,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)

# fit boosting regression tree on fixed effect in training sample to get MSE on 
# test sample for comparison
FE_boost_train <- gbm(as.numeric(FixedEffectReg) ~ . -SchoolID - RandEffectREEM - CNT, 
                      data = train_schools, 
                      distribution = "gaussian", 
                      n.trees = 5000, 
                      interaction.depth = 1,
                      shrinkage = 0.001,
                      n.minobsinnode = 3)
fe_boost_estimate = predict(FE_boost_train, 
                            newdata = test_schools, 
                            n.trees = 5000)
mean((fe_boost_estimate - test_schools$FixedEffectReg)^2)

# now fit a boosting regression tree on the fixed effects from the linear
# fixed effect model for comparison (same interaction depth)
FE_boost_opt <- gbm(as.numeric(FixedEffectReg) ~ . -SchoolID - RandEffectREEM - CNT, 
                data = school_df_ints, 
                distribution = "gaussian", 
                n.trees = 5000, 
                interaction.depth = 1,
                shrinkage = 0.001,
                n.minobsinnode = 3)

# Data for graphs

REEM_boost_varimp <- as.data.frame(relative.influence(REEM_boost_opt, sort. = TRUE))
REEM_boost_varimp <- REEM_boost_varimp %>%
  rename("RelInf" = "relative.influence(REEM_boost_opt, sort. = TRUE)")
REEM_boost_varimp <- head(REEM_boost_varimp,10)
REEM_boost_varimp <- REEM_boost_varimp %>% arrange(RelInf)

FE_boost_varimp <- as.data.frame(relative.influence(FE_boost_opt, sort. = TRUE))
FE_boost_varimp <- FE_boost_varimp %>%
  rename("RelInf" = "relative.influence(FE_boost_opt, sort. = TRUE)")
FE_boost_varimp <- head(FE_boost_varimp,10)
FE_boost_varimp <- FE_boost_varimp %>% arrange(RelInf)

# Variable Importance plots
par(mfrow = c(2,2))

dotchart(REEM_forest_varimp$IncNodePurity, labels = row.names(REEM_forest_varimp), 
         xlab = "Increase in Node Purity", main = "Stage 1: Regression Tree, Stage 2: Random Forest")

dotchart(FE_forest_varimp$IncNodePurity, labels = row.names(FE_forest_varimp), 
         xlab = "Increase in Node Purity", main = "Stage 1: FE Regression, Stage 2: Random Forest")

dotchart(REEM_boost_varimp$RelInf, labels = row.names(REEM_boost_varimp), 
         xlab = "Relative Influence", main = "Stage 1: Regression Tree, Stage 2: Boosting")

dotchart(FE_boost_varimp$RelInf, labels = row.names(FE_boost_varimp), 
         xlab = "Relative Influence", main = "Stage 1: FE Regression, Stage 2: Boosting")










