library(brms)
require(lme4)
library(lmerTest)
library(bayestestR)
library(ordinal)
library(dplyr)

set.seed(101)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dir_path = '../data/stats_analysis'

df_action_predction <- read.csv((file=file.path(dir_path, "loophole_prediction_df.csv")))

df_extended_action_prediction <- read.csv(file=file.path(dir_path, 'extended_loophole_prediction_df.csv'))

df_is_action_chosen_all <- bind_rows(df_action_predction, df_extended_action_prediction)
df_is_action_chosen_all$is_compliance_chosen <- ifelse(df_is_action_chosen_all$action == "compliance", 1, 0)
df_is_action_chosen_all$is_noncompliance_chosen <- ifelse(df_is_action_chosen_all$action == "noncompliance", 1, 0)
df_is_action_chosen_all$is_loophole_chosen <- ifelse(df_is_action_chosen_all$action == "loophole", 1, 0)

df_is_action_chosen_all$condition = relevel(factor(df_is_action_chosen_all$condition), ref="Low")
model_glmer <- glmer(formula = "is_noncompliance_chosen ~ condition + (1 | story) + (1 | subject)",
                     data=df_is_action_chosen_all, family=binomial(link="logit"))
print(summary(model_glmer))

model_glmer <- glmer(formula = "is_compliance_chosen ~ condition + (1 | story) + (1 | subject)",
                     data=df_is_action_chosen_all, family=binomial(link="logit"))
print(summary(model_glmer))

model_glmer <- glmer(formula = "is_loophole_chosen ~ condition + (1 | story) + (1 | subject)",
                     data=df_is_action_chosen_all, family=binomial(link="logit"))
print(summary(model_glmer))


# Analyze extended loophole evaluation data
data <- read.csv(file = file.path(dir_path, 'extended_loophole_evaluation_df.csv'))
data$response_num = factor(data$response_num, levels = c("1", "2", "3", "4"), ordered = TRUE)
data$behavior = relevel(factor(data$behavior), ref="Loophole")
trouble_model <- clmm(formula="response_num ~ condition + behavior + (1 + condition + behavior | subject) + (1 + condition + behavior | story)",
              data=data %>% filter(measure_type == "trouble"))
print(summary(trouble_model))

upset_model <- clmm(formula="response_num ~ condition + behavior + (1 + condition + behavior | subject) + (1 + condition + behavior | story)",
                    data=data %>% filter(measure_type == "upset"))
print(summary(upset_model))

funny_model <- clmm(formula="response_num ~ condition + behavior + (1 + condition + behavior | subject) + (1 + condition + behavior | story)",
                    data=data %>% filter(measure_type == "funny"))
print(summary(funny_model))

# Analyze extended loophole prediction data
df_is_action_chosen_extended <- df_extended_action_prediction
df_is_action_chosen_extended$is_compliance_chosen <- ifelse(df_extended_action_prediction$action == "compliance", 1, 0)
df_is_action_chosen_extended$is_noncompliance_chosen <- ifelse(df_extended_action_prediction$action == "noncompliance", 1, 0)
df_is_action_chosen_extended$is_loophole_chosen <- ifelse(df_extended_action_prediction$action == "loophole", 1, 0)

df_is_action_chosen_extended$condition = relevel(factor(df_is_action_chosen_extended$condition), ref="Low")
model_glmer <- glmer(formula = "is_noncompliance_chosen ~ condition + (1 | story) + (1 | subject)",
                     data=df_is_action_chosen_extended, family=binomial(link="logit"))
print(summary(model_glmer))

model_glmer <- glmer(formula = "is_compliance_chosen ~ condition + (1 | story) + (1 | subject)",
                     data=df_is_action_chosen_extended, family=binomial(link="logit"))
print(summary(model_glmer))

model_glmer <- glmer(formula = "is_loophole_chosen ~ condition + (1 | story) + (1 | subject)",
                     data=df_is_action_chosen_extended, family=binomial(link="logit"))
print(summary(model_glmer))
