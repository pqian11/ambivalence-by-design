from model import *
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import scipy.stats
matplotlib.rcParams["font.family"] = "arial"


path = 'data/understanding_data.csv'
df = pd.read_csv(path)

print('Data from {} participants are included in the analysis.'.format(len(set(df['ResponseId'].tolist()))))

condition_dict = json.load(open('data/stimuli_info/loophole_prediction_condition_dict.json', 'r'))
stories = condition_dict['stories']
behaviors = ['Comply', 'Loophole', 'NonComply']
actions = ['compliance', 'loophole', 'noncompliance']
power_relations = ['DOWN', 'EQUAL', 'UP']
options = ["didn't understand at all", 'may not have understood', 'may have understood', 'completely understood']

scores = [0, 1, 2, 3]
option2score = dict(zip(options, scores))

data = {}
for story in stories:
    data[story] = {}
    for behavior in behaviors:
        data[story][behavior] = []

for _, row in df.iterrows():
    response = row['response']
    story = row['story']
    condition = row['condition']
    power_relation, behavior = condition.split('_')
    measure_type = row['measure_type']
    response_num = row['response_num']

    if measure_type == 'understanding':
        data[story][behavior].append(response_num)


a_colors = ['#66A182', 'orange', "red"]  # Color coding for each action type
np.random.seed(11)

fig, axes = plt.subplots(1, 2, figsize=(7.8,3.5))
plt.subplots_adjust(wspace=0.3)

# Plot model output
ax = axes[0]
model_ys = [infer_understanding_of_L0(action, meaning_certainty=0.9, prosocial_prior_prob=0.85) for action in ['compliance', 'loophole', 'noncompliance']]

for j, behavior in enumerate(behaviors):
    ax.bar(j, model_ys[j], width=0.65, edgecolor=a_colors[j], facecolor='w', hatch='///')

ax.set_ylabel('P(m=intended)')
ax.set_xticks(np.arange(len(behaviors)))
ax.set_xticklabels([action.title() for action in actions])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 1.03)
ax.set_title('Model')

# Plot human data
ax = axes[1]
for j, behavior in enumerate(behaviors):
    ys = [np.mean(data[story][behavior]) for story in stories]
    y_sem = scipy.stats.sem(ys)
    ax.bar(j, np.mean(ys), yerr=1.96*y_sem, width=0.65, color=a_colors[j])
    y_mean = np.mean(ys)
    # print(y_mean, [y_mean-1.96*y_sem, y_mean+1.96*y_sem])
    ax.plot((np.random.random(len(ys))*2-1)*0.05 + j, ys, 'ko', mfc='none', alpha=0.2)
ax.set_ylim(0, 3.09)

# ax.set_ylabel('How much did the listener understand?')
ax.set_ylabel('Degree of understanding')
ax.set_xticks(np.arange(len(behaviors)))
ax.set_xticklabels([action.title() for action in actions])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('People')
plt.savefig('fig/loophole_understand_model_human_comparison.pdf', bbox_inches='tight')
plt.show()


# Display summary statistics
print('{:<10} {}'.format("Behavior", "Mean Understanding (95% CI)"))
for j, behavior in enumerate(behaviors):
    ratings = [np.mean(data[story][behavior]) for story in stories]
    rating_sem = scipy.stats.sem(ratings)
    rating_mean = np.mean(ratings)
    print('{:<10} {:.3f} ({:.3f}, {:.3f})'.format(behavior, rating_mean, rating_mean-1.96*rating_sem, rating_mean+1.96*rating_sem))
