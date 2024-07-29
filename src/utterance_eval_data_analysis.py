import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import scipy.stats
matplotlib.rcParams["font.family"] = "arial"

path = 'data/utterance_meaning_data.csv'

df = pd.read_csv(path)
condition_dict = json.load(open('data/stimuli_info/loophole_prediction_condition_dict.json'))
stories = condition_dict['stories']

data = {}
for story in stories:
    data[story] = []

for _, row in df.iterrows():
    response = row['response']
    story = row['story']
    scale2 = row['scale2']
    if scale2 == 'INTENDED':
        rating = response
    else:
        rating = 100 - response
    data[story].append(rating)


# Print out average rating of the intended meaning (0-100)
for story in stories:
    print("{:>15} {:.3f} (\u00B1{:.3f})".format(story, np.mean(data[story]), scipy.stats.sem(data[story])))


P_intended_list = np.array([np.mean(data[story]) for story in stories])/100
mean_rating = np.mean(P_intended_list)
rating_sem = scipy.stats.sem(P_intended_list)


# Plot a histogram of the meaning evaluation data
plt.figure(figsize=(4,2.5))
ax = plt.gca()
plt.hist(P_intended_list, alpha=0.3)
plt.xlim(0, 1)
plt.vlines(P_intended_list, 0, 0.6, color='k', linewidth=0.5)
plt.ylabel('Number of stories')
plt.xlabel('Likelihood of the intended meaning')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axvline(mean_rating, color='tab:blue')
plt.savefig('fig/P_intended_human_rating_hist.pdf', bbox_inches='tight')
plt.show()


# Plot a vertical boxplot of the meaning evaluation data
plt.figure(figsize=(2,3))
ax = plt.gca()
plt.plot((np.random.random(len(P_intended_list))*2-1)*0.01, P_intended_list, 'o', color='k', mfc='none', alpha=0.2)
plt.boxplot(P_intended_list, notch=False, sym='', positions=[0])
ax.set_xticks([])
ax.set_xlim(-0.3, 0.3)
ax.set_ylabel('Likelihood of\nthe intended meaning')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0,1.02)
plt.savefig('fig/P_intended_human_rating_boxplot.pdf', bbox_inches='tight')
plt.show()
