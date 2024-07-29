import pandas as pd
import numpy as np
import json


fname = 'study4_loophole_prediction_stories.csv'
df_story_info = pd.read_csv('data/stimuli_info/{}'.format(fname))

stories = []

response2type = {}
for _, row in df_story_info.iterrows():
    story = '_'.join(row['StoryName'].split('_')[1:])
    stories.append(story)
    
    compliance = row['Compliance'].strip()
    loophole = row['Loophole'].strip()
    noncompliance = row['Non-compliance'].strip()
    
    if story == 'house_parties':
        compliance = 'not have any parties.'
        loophole = 'host a party in the backyard.'
        noncompliance = 'host a party in the house.'
    elif story == 'listen_second':
        compliance = 'stop talking, take a breath, and listen to what he has to say.'
            
    response2type[story] = {}
    
    response2type[story][compliance] = 'compliance'
    response2type[story][loophole] = 'loophole'
    response2type[story][noncompliance] = 'noncompliance'


actions = ['compliance', 'loophole', 'noncompliance']
conditions = ['Low', 'High']
story_set = set(stories)


# Export data frame for the extended loophole prediction task (Study 4)
fname = 'extended_loophole_adult_prediction_data.csv'
df = pd.read_csv('data/{}'.format(fname))

data_rows = []

for row_idx, row in df.iterrows():
    story = row['story']

    condition = row['condition']

    response = row['response'].replace("‚Äú", "“").replace("‚Äù", "”")
    if response not in response2type[story]:
        print(story, response)
        print(response.replace("‚Äú", "“").replace("‚Äù", "”"))
        print(response2type[story])
        print()
    action = response2type[story][response]
    subject_id = row['ResponseId']

    data_rows.append([subject_id, condition, story, action])
        
extended_loophole_prediction_df = pd.DataFrame(data_rows, columns=['subject', 'condition', 'story', 'action'])
extended_loophole_prediction_df.to_csv('data/stats_analysis/extended_loophole_prediction_df.csv')


# Export data frame for the extended loophole evaluation task (Study 4)
fname = 'extended_loophole_adult_evaluation_data.csv'
df = pd.read_csv('data/{}'.format(fname))

fname = 'study4_loophole_evaluation_stories.csv'
df_story_info = pd.read_csv('data/stimuli_info/{}'.format(fname))

response2type = {}
for _, row in df_story_info.iterrows():
    story = '_'.join(row['StoryName'].split('_')[1:])
    
    compliance = row['Compliance'].strip()
    loophole = row['Loophole'].strip()
    noncompliance = row['Non-compliance'].strip()
    
    response2type[story] = {}
    
    response2type[story][compliance] = 'compliance'
    response2type[story][loophole] = 'loophole'
    response2type[story][noncompliance] = 'noncompliance'
    
actions = ['compliance', 'loophole', 'noncompliance']
behavior2action = {'Comply':'compliance', 'Loophole':'loophole', 'NonComply':'noncompliance'}
conditions = ['Low', 'High']

data_rows = []

for row_idx, row in df.iterrows():
    story = row['story']
    condition, behavior = row['condition'].split('_')
    response_num = row['response_num']
    response = row['response']
    measure_type = row['measure_type']
    subject_id = row['ResponseId']

    data_rows.append([subject_id, condition, story, behavior, measure_type, response, response_num+1])
    
extended_loophole_evaluation_df = pd.DataFrame(data_rows, columns=['subject', 'condition', 'story', 'behavior', 'measure_type', 'response', 'response_num'])
extended_loophole_evaluation_df.to_csv('data/stats_analysis/extended_loophole_evaluation_df.csv')


# Export data frame for the loophole prediction task (Bridgers et al., 2023)
fname = 'loophole_adult_prediction_data.csv'
df = pd.read_csv('data/{}'.format(fname))

data_rows = []

for row_idx, row in df.iterrows():
    story = row['story']
    
    condition = row['power_relation']
    goal = row['goal']
    
    # Only include trials with misaligned goals
    if goal == 'Aligned':
        continue
        
    action = row['response_type']
    
    subject_id = row['subject_id']

    data_rows.append([subject_id, condition, story, action])

loophole_prediction_df = pd.DataFrame(data_rows, columns=['subject', 'condition', 'story', 'action'])
loophole_prediction_df.to_csv('data/stats_analysis/loophole_prediction_df.csv')



# Export understanding measure data (Study 3)
path = 'data/understanding_data.csv'
df = pd.read_csv(path)

options = ["didn't understand at all", 'may not have understood', 'may have understood', 'completely understood']
scores = [0, 1, 2, 3]
option2score = dict(zip(options, scores))

understanding_df_rows_all = []

for _, row in df.iterrows():
    subject_id = row['ResponseId']
    response = row['response']
    story = row['story']
    condition = row['condition']
    power_relation, behavior = condition.split('_')
    
    response_lower = response.lower()
    if response_lower.endswith('funny') or response_lower.endswith('trouble'):
        continue

    cut = 2 if response.startswith('Yes') or response.startswith('No') else 1
    response = ' '.join(response.split()[cut:])

    # Normalize typo and stylistic variants in the choice sets
    if response == 'may have undertstood':
        response = 'may have understood'
    elif response == 'did not understand at all':
        response = 'didn\'t understand at all'
        
    score = option2score[response]
    
    # if behavior != 'Comply':
    #     understanding_df_rows.append([subject_id, story,  power_relation, behavior, score+1])
    
    understanding_df_rows_all.append([subject_id, story,  power_relation, behavior, score+1])

# understanding_df = pd.DataFrame(understanding_df_rows, columns=["subject", "story", "power", "behavior", "response"])
# understanding_df.to_csv('statistical_analysis/understanding_df.csv')

understanding_all_df = pd.DataFrame(understanding_df_rows_all, columns=["subject", "story", "power", "behavior", "response"])
understanding_all_df.to_csv('data/stats_analysis/understanding_all_df.csv')


# Export all measure data (Study 3)
path = 'data/understanding_data.csv'
df = pd.read_csv(path)

understanding_measure_options = ["didn't understand at all", 'may not have understood', 'may have understood', 'completely understood']
funny_measure_options = ["not funny", "a little bit funny", "funny", "very funny"]
trouble_measure_options = ["no trouble", "a little bit of trouble", "trouble", "a lot of trouble"]

scores = [0, 1, 2, 3]
understanding_measure_option2score = dict(zip(understanding_measure_options, scores))
funny_measure_option2score = dict(zip(funny_measure_options, scores))
trouble_measure_option2score = dict(zip(trouble_measure_options, scores))

study3_df_rows_all = []

for _, row in df.iterrows():
    subject_id = row['ResponseId']
    response = row['response']
    story = row['story']
    condition = row['condition']
    power_relation, behavior = condition.split('_')
    
    response_lower = response.lower()
    if response_lower.endswith('funny'):
        measure_type = 'funny'
        score = funny_measure_option2score[response_lower]
    elif response_lower.endswith('trouble'):
        measure_type = 'trouble'
        score = trouble_measure_option2score[response_lower]   
    else:
        cut = 2 if response.startswith('Yes') or response.startswith('No') else 1
        response = ' '.join(response.split()[cut:])

        # Normalize typo and stylistic variants in the choice sets
        if response == 'may have undertstood':
            response = 'may have understood'
        elif response == 'did not understand at all':
            response = 'didn\'t understand at all'
            
        score = understanding_measure_option2score[response]
        measure_type = 'understanding'
    
    study3_df_rows_all.append([subject_id, story,  power_relation, behavior, measure_type, response.title(), score+1])

study3_all_df = pd.DataFrame(study3_df_rows_all, columns=["subject", "story", "power", "behavior", "measure_type", "response", "response_num"])
study3_all_df.to_csv('data/stats_analysis/study3_all_df.csv')
