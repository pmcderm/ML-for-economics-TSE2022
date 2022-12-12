# Context: project for Machine Learning for Economics course.
# based on paper: https://doi.org/10.1016/j.ejor.2018.02.031
# replicate methods, and some extension
# Code Authors: Paul McDermott and Justin Standish-White

# Short Summary: We look at student level determinants of math test scores, 
# assuming a school level fixed effect. We extract the fixed effect as a 
# measure of school value-add, then use school level data to estimate
# top determinants of school value add

# data is public: https://www.oecd.org/pisa/data/2015database/

# import necessary libraries
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
setwd('/Users/paulmcdermott/My Drive/Master 2 Courses/Machine learning for economics /PISA/Data')

########################################
# Data Cleaning
########################################

# Read in data
student_level_df <- read_sas('student_PISA2015_main.sas7bdat')
school_level_df <- read_sas('school_PISA2015_main.sas7bdat')

# columns to keep in student level data
keep_columns_student <- c('CNTSCHID','CNT','ST004D01T','ESCS','IMMIG',
                          'ST071Q02NA','HISCED', 'ST076Q06NA','ST076Q11NA',
                          'DISCLISCI','TEACHSUP','MMINS','BELONG','MOTIVAT',
                          'ANXTEST','COOPERATE','EMOSUPS','CULTPOSS','HEDRES',
                          'PV3MATH')

# columns to keep in school level data
keep_columns_school <- c('SC064Q02TA', 'SC002Q01TA', 'SC002Q02TA', 'SC004Q02TA',
                         'SC012Q01TA','SC016Q01TA', 'SC017Q02NA', 'SC017Q06NA', 
                         'SC018Q01TA01', 'SC048Q02NA','SC048Q03NA', 'SC061Q01TA', 
                         'SC061Q03TA', 'SC061Q07TA', 'SC064Q03TA', 'SC013Q01TA', 
                         'SC017Q08NA', 'CNT', 'CNTSCHID', 'SC009Q01TA', 
                         'SC009Q13TA')

# subset to only France data, and only the above columns
fr_students_df <- subset(student_level_df,CNT == 'FRA',
                         select = keep_columns_student)
fr_school_df <- subset(school_level_df,CNT == 'FRA',
                       select = keep_columns_school)

# Student data cleaning

# SAS includes a lot of meta information about the tables, which interferes
# with some methods in R. write then read to csv is a quick solution.
write.csv(fr_students_df, file = "students_france.csv")
students_df <- read.csv("students_france.csv", header=T)

# omit rows with NA, and get rid of indexx
students_df <- na.omit(students_df)
students_df <- students_df[,-1]

# rename columns for easier comprehnsion
students_df <- students_df %>%
  rename("SchoolID" = "CNTSCHID",
         "Gender" = "ST004D01T",
         "MathHW"="ST071Q02NA",
         "ParentalEduc"="HISCED",
         "VideoGames"="ST076Q06NA",
         "Sports"="ST076Q11NA",
         "DiscClimate"="DISCLISCI",
         "math_PV" = "PV3MATH")

# normalize dependent variable
students_df$math_PV <- scale(students_df$math_PV)

# school data cleaning
# same trick as before to get rid of meta information
write.csv(fr_school_df, file = "school_france.csv")
school_df <- read.csv("school_france.csv", header=T)

# rename columns for easier comprehension
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

# get rid of index
school_df <- school_df[,-1]

# data transformations
# total students
school_df$NmbStudents <- school_df$EnrollmentBoys + school_df$EnrollmentGirls
# computers per student
school_df$CompsPerStudent <- school_df$NumComputers/school_df$NmbStudents
# student teacher ratio
school_df$StudentTeacherRatio <- school_df$NmbStudents/school_df$TotalTeachersFT

# some columns used for transformations are no longer needed
school_df <- select(school_df, -c("EnrollmentBoys", "EnrollmentGirls", 
                                  "NumComputers","TotalTeachersFT"))

# omit NA
school_df <- na.omit(school_df)

#######################
# REEM Tree Replication
#######################
# the paper uses REEM trees to estimate student level determinants of math test
# scores, and also to estimate the school value add as a random effect

set.seed(22234110)
# build regression tree math scores with school random effects
treeREEM <- REEMtree(math_PV ~ Gender + ESCS + factor(IMMIG) + MathHW + 
                       ParentalEduc + VideoGames + Sports + DiscClimate + 
                       TEACHSUP + MMINS + BELONG + MOTIVAT + ANXTEST + 
                       COOPERATE + EMOSUPS + CULTPOSS + HEDRES,
                  data = students_df, random = ~1|SchoolID)
# visualize tree
plot(treeREEM)
text(treeREEM, pretty = 0)

# extract random effects, bind school id
treeREints <- as.data.frame(treeREEM$RandomEffects)
treeREints <- cbind(SchoolID = rownames(treeREints), treeREints)

# name random effects column
treeREints <- treeREints %>%
  rename("RandEffectREEM" = "(Intercept)")

# recast school id as integer
treeREints$SchoolID <- as.integer(treeREints$SchoolID)

# add random effects estimate to the school data with a join
school_df_ints <- inner_join(school_df, treeREints, by = 'SchoolID')

#######################
# FE Regression
#######################
# we extend the paper by also providing a linear fixed effect estimation of
# the student level determinants of math test scores, and also of the
# school value add, this time as a fixed effect
regFE <- plm(math_PV ~ factor(Gender) + ESCS + factor(IMMIG) + MathHW + 
               ParentalEduc + VideoGames + Sports + DiscClimate + TEACHSUP + 
               MMINS + BELONG + MOTIVAT + ANXTEST + COOPERATE + EMOSUPS + 
               CULTPOSS + HEDRES,
             data = students_df, model = c("within"), index = "SchoolID")

# get regression summary and get latex code for table
summary(regFE)
stargazer(regFE, type="latex")

# extract fixed effects, bind to school id
regFEints <- as.data.frame(fixef(regFE))
regFEints <- cbind(SchoolID = rownames(regFEints), regFEints)

# rename fixed effects
regFEints <- regFEints %>%
  rename("FixedEffectReg" = "fixef(regFE)")

# recast school id as integer
regFEints$SchoolID <- as.integer(regFEints$SchoolID)

# add fixed effects estimates to school level data with join
school_df_ints <- inner_join(school_df_ints, regFEints, by = 'SchoolID')

# comparison: fixed effects from the two are highly correlated
cor(school_df_ints$FixedEffectReg, school_df_ints$RandEffectREEM)

# clear relationship. Reg FE ones are not centered at 0 as model has no constant
# tree essentially has a constant 
plot(school_df_ints$FixedEffectReg, school_df_ints$RandEffectREEM)

################
# Random Forests
################
# next step is to estimate determinants of school value add, our fixed or random
# effect. the paper uses boosted regression trees. we replicate below, but first
# extend with random forests. 

# mtry i.e. the number of covariates kept by the random forest at each split,
# is a hyperparameter to be tuned. Method: k-fold cross-validation

# REEM Tree k-fold cross-validation
# Set the number of folds for cross-validation
nfolds <- 10

# Set the range of mtry values to test. small data set allows us to test all
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
                 trControl = trainControl(method = "cv", number = nfolds, 
                                          repeats = 3),
                 tuneGrid = expand.grid(mtry = mtry_values))
# print optimal mtry according to this round of cv
print(rf_reem_cv$bestTune$mtry)



# use tuned mtry to build "optimal" random forest on full data
REEM_forest_opt <- randomForest(RandEffectREEM ~ . -SchoolID - FixedEffectReg 
                                - CNT, 
                                data=school_df_ints,
                                mtry = rf_reem_cv$bestTune$mtry,
                                importance = TRUE)

# FE Tree cross-validation
# y data will be the fixed effect column this time
y_school_FE_df <- as.numeric(school_df_ints$FixedEffectReg)

# Perform cross-validation
rf_fe_cv <- train(x = x_school_df, y = y_school_FE_df,
                    method = "rf",
                    trControl = trainControl(method = "cv", number = nfolds),
                    tuneGrid = expand.grid(mtry = mtry_values))
# print optimal mtry according to this round of cv
print(rf_fe_cv$bestTune$mtry)

# use tuned mtry to build random forest on full data, this time with fixed
# effects from linear fe model
FE_forest_opt <- randomForest(FixedEffectReg ~ . -SchoolID - RandEffectREEM - 
                                CNT, 
                              data=school_df_ints,
                              mtry = rf_fe_cv$bestTune$mtry,
                              importance = TRUE)

# Data for graphs comparing variable importance for each stage 1 effects type
# (plots made outside of R)

REEM_forest_varimp <- as.data.frame(importance(REEM_forest_opt))
REEM_forest_varimp <- REEM_forest_varimp %>% arrange(desc(IncNodePurity))
REEM_forest_varimp <- head(REEM_forest_varimp,10)
REEM_forest_varimp <- REEM_forest_varimp %>% arrange(IncNodePurity)

FE_forest_varimp <- as.data.frame(importance(FE_forest_opt))
FE_forest_varimp <- FE_forest_varimp %>% arrange(desc(IncNodePurity))
FE_forest_varimp <- head(FE_forest_varimp,10)
FE_forest_varimp <- FE_forest_varimp %>% arrange(IncNodePurity)

# Partial dependence plots allow us to see the effect of a single covariate
# while "averaging out" the other covariates. we look at % disadvantaged students.
par(mfrow = c(1,2))
partialPlot(REEM_forest_opt, pred.data = school_df_ints, 
            x.var = "DisadvStudsNum", main = "Stage 1: Regression Tree", 
            xlab = "% Disadvantaged Students")
partialPlot(FE_forest_opt, pred.data = school_df_ints, x.var = "DisadvStudsNum", 
            main = "Stage 1: FE Regression", xlab = "% Disadvantaged Students")

################
# Boosted regression trees
################
# like in the paper, we use boosted regression trees to examine determinants of
# school value add. An extension: tuning interaction depth

# set values of interaction depth to test, we set max as 12 to save on compute
interaction_depth_range <- seq(1,12,1)
boosted_nfolds <- 10

# make grid for cv. only tuning interaction depth, rest is fixed. 
gbm_grid <-  expand.grid(interaction.depth = interaction_depth_range, 
                        n.trees = 3000, 
                        shrinkage = 0.001,
                        n.minobsinnode = 3)

# set train control, we're using repeated cv. we set repeat to 1 to save on
# compute for this exercise.
gbm_tr_control <- trainControl(method = "repeatedcv",
                               number = boosted_nfolds,
                               repeats = 1)

# Perform cross-validation for each interaction depth
boosted_reem_cv <- train(x = x_school_df, y = y_school_REEM_df,
                         method = "gbm",
                         trControl = gbm_tr_control,
                         tuneGrid = gbm_grid,
                         verbose = FALSE)
print(boosted_reem_cv$bestTune$interaction.depth)

# make boosting regression tree with tuned interaction depth, on full data
REEM_boost_opt = gbm(RandEffectREEM ~ . -SchoolID - FixedEffectReg - CNT, 
                    data = school_df_ints, 
                    distribution = "gaussian", 
                    n.trees = 3000, 
                    interaction.depth = boosted_reem_cv$bestTune$interaction.depth,
                    shrinkage = 0.001,
                    n.minobsinnode = 3)

# now cross validate interaction depth on the fixed effect tree
boosted_fe_cv <- train(x = x_school_df, y = y_school_FE_df,
                       method = "gbm",
                       trControl = gbm_tr_control,
                       tuneGrid = gbm_grid,
                       verbose = FALSE)
print(boosted_fe_cv$bestTune$interaction.depth)

# now fit a boosting regression tree on the fixed effects from the linear
# fixed effect model for comparison (same interaction depth)
FE_boost_train <- gbm(as.numeric(FixedEffectReg) ~ . -SchoolID - RandEffectREEM
                      - CNT, 
                      data = school_df_ints, 
                      distribution = "gaussian", 
                      n.trees = 5000, 
                      interaction.depth = boosted_fe_cv$bestTune$interaction.depth,
                      shrinkage = 0.001,
                      n.minobsinnode = 3)

# Note: in practice, we would want to tune more than just interaction
# depth, but this would add significant compute time for this exercise, so we
# tune just interaction to display the methodology

# Data for graphs

REEM_boost_varimp <- as.data.frame(relative.influence(REEM_boost_opt,
                                                      sort. = TRUE))
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
         xlab = "Increase in Node Purity", 
         main = "Stage 1: Regression Tree, Stage 2: Random Forest")

dotchart(FE_forest_varimp$IncNodePurity, labels = row.names(FE_forest_varimp), 
         xlab = "Increase in Node Purity", 
         main = "Stage 1: FE Regression, Stage 2: Random Forest")

dotchart(REEM_boost_varimp$RelInf, labels = row.names(REEM_boost_varimp), 
         xlab = "Relative Influence", 
         main = "Stage 1: Regression Tree, Stage 2: Boosting")

dotchart(FE_boost_varimp$RelInf, labels = row.names(FE_boost_varimp), 
         xlab = "Relative Influence", 
         main = "Stage 1: FE Regression, Stage 2: Boosting")










