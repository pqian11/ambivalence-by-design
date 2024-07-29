library(brms)
require(lme4)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dir_path = '../data/stats_analysis'

df_action_predction <- read.csv((file=file.path(dir_path, "loophole_prediction_df.csv")))

df_extended_action_prediction <- read.csv(file=file.path(dir_path, 'extended_loophole_prediction_df.csv'))

# Only analyze trials for the commonly shared stories across Bridgers et al. and Study 4b
shared_stories <- c("some_paper", "do_dishes", "help_laundry", "all_licorice", 
                    "shoes_away", "walk_dog", "some_weeds", "sip_soda", "read_chapter", 
                    "move_couch", "bed_time", "phone_out", "start_early", "trash_out", 
                    "come_class", "phone_down", "mow_lawn", "check_bathroom", "join_us", 
                    "no_facebook", "bounce_kitchen", "that_brownie", "drinking_steve", 
                    "house_parties", "lantern_off", "use_sugar", "no_phones", "listen_second")
  
df_action_predction <- df_action_predction %>% filter(story %in% shared_stories)
df_extended_action_prediction <- df_extended_action_prediction %>% filter(story %in% shared_stories)

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
