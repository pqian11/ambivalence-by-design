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


# Build data dictionary
data = {}
for behavior in behaviors:
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
      

# Display mean trouble rating for each behavior across power relations
print('People\'s average rating (0-3 four-point scale) on the trouble measure for each type of behaviors:\n')
print('{:<6} {:<10} {:<15}'.format("Power", "Behavior", "Trouble"))
print('-'*25)
for power_relation in power_relations:
    for behavior in behaviors:
        print('{:<6} {:<10} {:.3f}'.format(power_relation, behavior, np.mean([np.mean(data[behavior][power_relation][story]) for story in stories])))


# Plot the comparison between model and human judgment data
a_colors = ['#66A182', 'orange', "red"]
    
np.random.seed(11)
fig, axes = plt.subplots(1, 2, figsize=(7,3.5))
plt.subplots_adjust(wspace=0.3)

# Plot model predictions
ax = axes[0]
model_noncooperative_intent_probs = dict(zip(behaviors, [0.01557, 0.348, 0.463]))
mean_noncomply_trouble_rating = np.mean([[np.mean(data['NonComply'][power_relation][story]) for power_relation in power_relations] for story in stories])
trouble_scale = mean_noncomply_trouble_rating/model_noncooperative_intent_probs['NonComply']

for j, behavior in enumerate(behaviors):
    ax.bar(j, model_noncooperative_intent_probs[behavior]*trouble_scale, width=0.65, edgecolor=a_colors[j], facecolor='w', hatch='///')

ax.set_ylabel('Social cost')
ax.set_xticks(np.arange(len(behaviors)))
ax.set_xticklabels(behaviors)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 3)
ax.set_title('Model')

# Plot human behavioral data
ax = axes[1]

for j, behavior in enumerate(behaviors):
    ax.bar(j, np.mean([[np.mean(data[behavior][power_relation][story]) for power_relation in power_relations] for story in stories]), 
           color=a_colors[j], yerr=1.96*scipy.stats.sem([np.mean([np.mean(data[behavior][power_relation][story]) for power_relation in power_relations]) for story in stories]),
           width=0.65)
        
for j, behavior in enumerate(behaviors):
    ax.plot((np.random.random(len(stories))*2-1)*0.05 + j, [np.mean([np.mean(data[behavior][power_relation][story]) for power_relation in power_relations]) for story in stories], 'o', color='k', mfc='none', alpha=0.15)
        

ax.set_ylim(0,3)
ax.set_title('People')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(len(behaviors)))
ax.set_xticklabels(behaviors)
plt.ylabel('Trouble rating')
plt.savefig('fig/loophole_eval_trouble_model_human_comparison.pdf', bbox_inches='tight')
plt.show()        
