require(lme4)
library(ordinal)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dir_path = '../data/stats_analysis'

# data <- read.csv(file = file.path(dir_path, 'understanding_all_df.csv'))
# data$response= factor(data$response, levels = c("1", "2", "3", "4"), ordered = TRUE)
# data$behavior = relevel(factor(data$behavior), ref="Loophole")
# model <- clmm(formula="response ~ power + behavior + (1 + power + behavior | subject) + (1 + power + behavior | story)",
#                data=data)
# print(summary(model))


data_all <- read.csv(file = file.path(dir_path, 'study3_all_df.csv'))

# Analysis of understanding measure
df_understanding <- data_all %>% filter(measure_type == "understanding")
df_understanding$response_num = factor(df_understanding$response_num, levels = c("1", "2", "3", "4"), ordered = TRUE)
df_understanding$behavior = relevel(factor(df_understanding$behavior), ref="Loophole")
model_understanding <- clmm(formula="response_num ~ power + behavior + (1 + power + behavior | subject) + (1 + power + behavior | story)",
              data=df_understanding)
print(summary(model_understanding))

# Analysis of trouble measure
df_trouble <- data_all %>% filter(measure_type == "trouble")
df_trouble$response_num = factor(df_trouble$response_num, levels = c("1", "2", "3", "4"), ordered = TRUE)
df_trouble$behavior = relevel(factor(df_trouble$behavior), ref="Loophole")
model_trouble <- clmm(formula="response_num ~ power + behavior + (1 + power + behavior | subject) + (1 + power + behavior | story)",
                            data=df_trouble)
print(summary(model_trouble))

# Analysis of funny measure
df_funny <- data_all %>% filter(measure_type == "funny")
df_funny$response_num = factor(df_funny$response_num, levels = c("1", "2", "3", "4"), ordered = TRUE)
df_funny$behavior = relevel(factor(df_funny$behavior), ref="Loophole")
model_funny <- clmm(formula="response_num ~ power + behavior + (1 + power + behavior | subject) + (1 + power + behavior | story)",
                      data=df_funny)
print(summary(model_funny))