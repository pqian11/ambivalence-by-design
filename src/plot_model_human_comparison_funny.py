from model import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
import scipy.stats
matplotlib.rcParams['font.family'] = "arial"

fname = 'loophole_adult_evaluation_data.csv'

df = pd.read_csv('data/{}'.format(fname))

stories = list(set(df['story']))
power_relations = ['DOWN', 'EQUAL', 'UP']
behaviors = ['Comply', 'Loophole', 'NonComply']
actions = ['compliance', 'loophole', 'noncompliance']

# stories = list(set(df['story']))
# power_relations = list(set(df['power_relation']))
# behaviors = ['Comply','Loophole','NonComply']

funny_data = {}

for behavior in behaviors:
    funny_data[behavior] = {}
    for power_relation in power_relations:
        funny_data[behavior][power_relation] = {}
        for story in stories:
            funny_data[behavior][power_relation][story] = []
            
for row_idx, row in df.iterrows():
    story = row['story']
    power_relation = row['power_relation']
    behavior = row['behavior']
    
    if row['measure_type'] == 'funny':
        funny_data[behavior][power_relation][story].append(row['response_num'])
        

# Display mean trouble rating for each behavior across power relations
print('{:<6} {:<10} {:<15}'.format("Power", "Behavior", "Funny"))
print('-'*25)
for power_relation in power_relations:
    for behavior in behaviors:
        print('{:<6} {:<10} {:.3f}'.format(power_relation, behavior, np.mean([np.mean(funny_data[behavior][power_relation][story]) for story in stories])))


# Plot model and human data
a_colors = ['#66A182', 'orange', "red"]
    
np.random.seed(11)

fig, axes = plt.subplots(1, 2, figsize=(7.8,3.5))
plt.subplots_adjust(wspace=0.3)

ax = axes[0]
model_ys = [calc_strength_of_humor_as_distinctiveness_support(action, meaning_certainty=0.9, prosocial_prior_prob=0.85) for action in actions] # Distinctiveness measure for Comply, Loophole, and NonComply based on the model

for j, behavior in enumerate(behaviors):
    ax.bar(j, model_ys[j], width=0.65, edgecolor=a_colors[j], facecolor='w', hatch='///')

ax.set_ylabel('Distinctive support\nfor ambivalent social intent')
ax.set_xticks(np.arange(len(behaviors)))
ax.set_xticklabels([action.title() for action in actions])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0,1.5)
ax.set_title('Model')

ax = axes[1]

for j, behavior in enumerate(behaviors):
    ax.bar(j, np.mean([[np.mean(funny_data[behavior][power_relation][story]) for power_relation in power_relations] for story in stories]), 
           color=a_colors[j],  yerr=1.96*scipy.stats.sem([np.mean([np.mean(funny_data[behavior][power_relation][story]) for power_relation in power_relations]) for story in stories]),
           width=0.65)        
        
for j, behavior in enumerate(behaviors):
    ax.plot((np.random.random(len(stories))*2-1)*0.05 + j, [np.mean([np.mean(funny_data[behavior][power_relation][story]) for power_relation in power_relations]) for story in stories], 'o', color='k', mfc='none', alpha=0.15)
        
ax.set_ylim(0, 1.5)
ax.set_title('People')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(3))
ax.set_xticklabels([action.title() for action in actions])
plt.ylabel('Funny rating')
plt.savefig('fig/loophole_eval_funny_model_human_comparison.pdf', bbox_inches='tight')
plt.show()

        