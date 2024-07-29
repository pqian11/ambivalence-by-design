from model import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import json
import scipy.stats
matplotlib.rcParams['font.family'] = "arial"


actions = ['compliance', 'loophole', 'noncompliance']
a_colors = ['#66A182', 'orange', "red"]


# Load participants' trouble ratings in the action evaluation task from Bridgers et al.
condition_dict = json.load(open('data/stimuli_info/loophole_prediction_condition_dict.json', 'r'))
stories = condition_dict['stories']
power_relations = condition_dict['power_relations']
behaviors = condition_dict['behaviors']

df = pd.read_csv('data/loophole_adult_evaluation_data.csv')

# Note that people's choices were coded with a slightly different set of words in the dataframe.
response_behaviors = ['Comply','Loophole','NonComply'] 
data = {}

for behavior in response_behaviors:
    data[behavior] = {}
    for power_relation in power_relations:
        data[behavior][power_relation] = {}
        for story in stories:
            data[behavior][power_relation][story] = []

for row_idx, row in df.iterrows():
    story = row['story']
    power_relation = row['power_relation']
    behavior = row['behavior']
    
    if row['measure_type'] == 'trouble':
        data[behavior][power_relation][story].append(row['response_num'])


social_cost_dict = {}

get_pretty_behavior_name = {'Comply':'compliance', 'Loophole':'loophole', 'NonComply':'noncompliance'}

for power_relation in power_relations:
    social_cost_dict[power_relation] = {}
    for story in stories:
        social_cost_dict[power_relation][story] = {}
        for behavior in response_behaviors:
            social_cost_dict[power_relation][story][get_pretty_behavior_name[behavior]] = np.mean(data[behavior][power_relation][story])


# Load participant's action choices in the action prediction task from Bridgers et al.
df = pd.read_csv('data/loophole_adult_prediction_data.csv')

a_freq_dict = {}

for power_relation in power_relations:
    a_freq_dict[power_relation] = {}
    for story in stories:
        a_freq_dict[power_relation][story] = {}
        for behavior in behaviors:
            a_freq_dict[power_relation][story][behavior] = 0

for row_idx, row in df.iterrows():
    story = row['story']
    power_relation = row['power_relation']
    behavior = row['response_type']
    
    if row['goal'] == 'Misaligned':
        a_freq_dict[power_relation][story][behavior] += 1


# Reduce the trouble ratings across stories to get average social cost for each behavior across power relations;
# cost_of_nonobedience_all stores the mean trouble ratings for the non-compliance behavior across DOWN, EQUAL, UP relations.
social_cost_data = [[[social_cost_dict[power_relation][story][behavior] for behavior in behaviors] for story in stories] for power_relation in power_relations]
C_mean_all = np.mean(social_cost_data, axis=1)
cost_of_nonobedience_all = [C_mean_all[i][-1] for i in range(len(power_relations))]

# Normalize the frequency count to a probability distribution within each story-power relation combination;
# For each power relation condition, compute the action probability distribution averaged across stories.
a_freq_data = [[[a_freq_dict[power_relation][story][behavior] for behavior in behaviors] for story in stories] for power_relation in power_relations]
a_prob_data = np.zeros((len(power_relations), len(stories), len(behaviors)))
for i in range(len(power_relations)):
    for j in range(len(stories)):
        a_prob_data[i, j] = normalize(a_freq_data[i][j])
a_prob_mean_all = np.mean(a_prob_data, axis=1)
a_prob_sem_all = scipy.stats.sem(a_prob_data, axis=1)


# Load human behavioral data from Study 4a (action evaluation)
fname = 'study4_loophole_evaluation_stories.csv'
df_study4a_story_info = pd.read_csv('data/stimuli_info/{}'.format(fname))
study4_stories = ['_'.join(row['StoryName'].split('_')[1:]) for _, row in df_study4a_story_info.iterrows()]

behavior2action = {'Comply':'compliance', 'Loophole':'loophole', 'NonComply':'noncompliance'}
situations = ['Low', 'High']

fname = 'extended_loophole_adult_evaluation_data.csv'
df = pd.read_csv('data/{}'.format(fname))

data = {}

for story in study4_stories:
    data[story] = {}
    for situation in situations:
        data[story][situation] = {}
        for action in actions:
            data[story][situation][action] = []

data_all = {}

measure_types = ['funny', 'upset', 'trouble']

for measure_type in measure_types:
    data_all[measure_type] = {}
    for story in study4_stories:
        data_all[measure_type][story] = {}
        for situation in situations:
            data_all[measure_type][story][situation] = {}
            for action in actions:
                data_all[measure_type][story][situation][action] = []

for row_idx, row in df.iterrows():
    story = row['story']
    situation, behavior = row['condition'].split('_')
    response = row['response_num']
    measure_type = row['measure_type']
    data_all[measure_type][story][situation][behavior2action[behavior]].append(response)

extended_social_cost_dict = data_all['trouble']
extended_social_cost_data = [[[np.mean(extended_social_cost_dict[story][situation][behavior]) for behavior in behaviors] for story in study4_stories] for situation in situations]
extended_C_mean_all = np.mean(extended_social_cost_data, axis=1)
# print(extended_C_mean_all)


# Load human behavioral data from Study 4b (action prediction)
fname = 'study4_loophole_prediction_stories.csv'
df_study4b_story_info = pd.read_csv('data/stimuli_info/{}'.format(fname))

response2type = {}
for _, row in df_study4b_story_info.iterrows():
    story = '_'.join(row['StoryName'].split('_')[1:])

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

fname = 'extended_loophole_adult_prediction_data.csv'
df = pd.read_csv('data/{}'.format(fname))

extended_a_freq_dict = {}
for story in study4_stories:
    extended_a_freq_dict[story] = {}
    for situation in situations:
        extended_a_freq_dict[story][situation] = {}
        for action in actions:
            extended_a_freq_dict[story][situation][action] = 0

for row_idx, row in df.iterrows():
    story = row['story']

    situation = row['condition']

    response = row['response'].replace("‚Äú", "“").replace("‚Äù", "”")
    action = response2type[story][response]
    extended_a_freq_dict[story][situation][action] += 1


extended_a_freq_data = [[[extended_a_freq_dict[story][situation][behavior] for behavior in behaviors] for story in study4_stories] for situation in situations]

extended_a_prob_data = np.zeros((len(situations), len(study4_stories), len(behaviors)))

for i in range(len(situations)):
    for j in range(len(study4_stories)):
        extended_a_prob_data[i, j] = normalize(extended_a_freq_data[i][j])
        
extended_a_prob_mean_all = np.mean(extended_a_prob_data, axis=1)
extended_a_prob_sem_all = scipy.stats.sem(extended_a_prob_data, axis=1)


C_mean_list = [extended_C_mean_all[0]] + list(C_mean_all) + [extended_C_mean_all[1]]
a_prob_mean_list = [extended_a_prob_mean_all[0]] + list(a_prob_mean_all) + [extended_a_prob_mean_all[1]]
a_prob_sem_list = [extended_a_prob_sem_all[0]] + list(a_prob_sem_all) + [extended_a_prob_sem_all[1]]

# print(C_mean_list)
# print(a_prob_mean_list)

# Plot comparison between model and people
action2linestyle = {'compliance':'-', 'loophole':'-.', 'noncompliance':'dotted'}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.2))

a_probs_list = []

rho = 0.3
alpha = 1.5
meaning_certainty = 0.9
prosocial_prior_prob = 0.85

U_S_intended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, -1, -1])
U_S_unintended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, 1, -1])

U_S = make_dict(actions, np.array([U_S_intended[action]*meaning_certainty + U_S_unintended[action]*(1 - meaning_certainty) for action in actions]))
U_L = make_dict(actions, [-1, 0.7, 1])

L0_noncooperative_intent_given_action_probs = [infer_noncooperative_intent_of_L0(action, meaning_certainty, prosocial_prior_prob) for action in actions]

cost_of_nonobedience_list = np.arange(0, 2.5, 0.01)
for cost_of_nonobedience in cost_of_nonobedience_list:
    lambda_scale = cost_of_nonobedience/L0_noncooperative_intent_given_action_probs[-1]
    C_social = make_dict(actions, [L0_noncooperative_intent_given_action_probs[0]*lambda_scale, 
                                L0_noncooperative_intent_given_action_probs[1]*lambda_scale, 
                                L0_noncooperative_intent_given_action_probs[2]*lambda_scale])
    a_dist = get_action_dist(get_combined_U(U_S, U_L, rho, C_social, actions), actions, alpha=alpha)
    a_probs_list.append([a_dist[a] for a in actions])

ax = axes[0]
for i, action in enumerate(actions):
    ax.plot(cost_of_nonobedience_list,  [a_probs[i] for a_probs in a_probs_list], ls=action2linestyle[action], color=a_colors[i], label=action.title())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, np.max(cost_of_nonobedience_list))
ax.set_ylim(ymin=0, ymax=0.68)
ax.text(0.6, 0.06, r"$\rho="+"{:.1f}".format(rho)+r"$  $\alpha="+"{:.1f}".format(alpha)+r"$", transform=ax.transAxes)

ax.legend(loc="center", bbox_to_anchor=(1.05, 1.2), ncol=3)
ax.set_xlabel('Expected penalty for non-compliance')
ax.set_ylabel('P(a)')

ax.set_title('Model')

ax = axes[1]
for j in range(len(behaviors)):
    plt.errorbar([C_mean_list[n][-1] for n in range(len(C_mean_list))], [a_prob_mean_list[n][j] for n in range(len(a_prob_mean_list))], yerr=[a_prob_sem_list[n][j]*1.96 for n in range(len(a_prob_mean_list))], 
        marker='None', ls=action2linestyle[behaviors[j]], color=a_colors[j])


markers = ['o', 'v', 's', '^', 'x']
for j in range(len(behaviors)):
    for n in range(len(a_prob_mean_list)):
        plt.errorbar(C_mean_list[n][-1], a_prob_mean_list[n][j], yerr=a_prob_sem_list[n][j]*1.96, 
            marker=markers[n], ls=action2linestyle[behaviors[j]], color=a_colors[j])

conditions = ['LOW COST', 'DOWN', 'EQUAL', 'UP', 'HIGH COST']
condition_patches = []
for n, marker in enumerate(markers):
    condition_patches.append(Line2D([0], [0], color='gray', marker=marker, ls='None', markersize=4, label=conditions[n]))
plt.legend(handles=condition_patches, bbox_to_anchor=(0.02, 0.02), loc='lower left', ncol=2, fontsize=7, handletextpad=0.1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(ymin=0, ymax=0.68)
ax.set_xlabel('Trouble rating of non-compliance')
ax.set_ylabel('Frequency')
ax.set_title('People')

plt.savefig('fig/model_human_a_pred_comparison.pdf', bbox_inches='tight')
plt.show()
