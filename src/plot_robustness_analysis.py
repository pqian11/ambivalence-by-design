from model import *
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams['font.family'] = "arial"


actions = ['compliance', 'loophole', 'noncompliance']

# Load human behavioral data from Bridgers et al. for fitting lambdas, rho, and alpha in the model.
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


# Load human behavioral data from Study 4a
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


# Load human behavioral data from Study 4b
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
# print(extended_a_freq_data[1][:3])

extended_a_prob_data = np.zeros((len(situations), len(study4_stories), len(behaviors)))

for i in range(len(situations)):
    for j in range(len(study4_stories)):
        extended_a_prob_data[i, j] = normalize(extended_a_freq_data[i][j])
        
extended_a_prob_mean_all = np.mean(extended_a_prob_data, axis=1)
extended_a_prob_sem_all = scipy.stats.sem(extended_a_prob_data, axis=1)
# print(extended_a_prob_mean_all)


def cost_pattern_MSE(meaning_certainty, prosocial_prior_prob, mean_costs):
    actions = ['compliance', 'loophole', 'noncompliance']

    L0_noncooperative_intent_given_action_probs = [infer_noncooperative_intent_of_L0(action, meaning_certainty, prosocial_prior_prob) for action in actions]

    human_loophole_over_noncompliance_ratio = mean_costs[1]/mean_costs[2]
    human_compliance_over_noncompliance_ratio = mean_costs[0]/mean_costs[2]

    model_loophole_over_noncompliance_ratio = L0_noncooperative_intent_given_action_probs[1]/L0_noncooperative_intent_given_action_probs[2]
    model_compliance_over_noncompliance_ratio = L0_noncooperative_intent_given_action_probs[0]/L0_noncooperative_intent_given_action_probs[2]

    return np.mean(np.square([model_loophole_over_noncompliance_ratio - human_loophole_over_noncompliance_ratio, model_compliance_over_noncompliance_ratio - human_compliance_over_noncompliance_ratio]))
    # return np.mean(np.square(np.array(L0_noncooperative_intent_given_action_probs) - np.array(mean_costs)))


def understanding_pattern_MSE(meaning_certainty, prosocial_prior_prob, human_understanding_probs):
    actions = ['compliance', 'loophole', 'noncompliance']

    model_understanding_probs = np.array([infer_understanding_of_L0(action, meaning_certainty, prosocial_prior_prob) for action in actions])

    return np.mean(np.square(model_understanding_probs - human_understanding_probs))


def action_prob_pattern_MSE(meaning_certainty, prosocial_prior_prob, rho, alpha):
    actions = ['compliance', 'loophole', 'noncompliance']
    U_S_intended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, -1, -1])
    U_S_unintended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, 1, -1])

    L0_noncooperative_intent_given_action_probs = [infer_noncooperative_intent_of_L0(action, meaning_certainty, prosocial_prior_prob) for action in actions]

    lambdas = []
    for n, cost_of_nonobedience in enumerate(cost_of_nonobedience_all):
        lambdas.append(cost_of_nonobedience/L0_noncooperative_intent_given_action_probs[-1]) # C_social = lambda*P_noncooperative; lambda = C_social/P_noncooperative

    U_S = make_dict(actions, np.array([U_S_intended[action]*meaning_certainty + U_S_unintended[action]*(1 - meaning_certainty) for action in actions]))
    U_L = make_dict(actions, np.array([-1, 0.7, 1])) 

    mse_list = []
    a_probs_list = []
    
    for n, cost_of_nonobedience in enumerate(cost_of_nonobedience_all):
        C_social = make_dict(actions, [L0_noncooperative_intent_given_action_probs[0]*lambdas[n], 
                                    L0_noncooperative_intent_given_action_probs[1]*lambdas[n], 
                                    L0_noncooperative_intent_given_action_probs[2]*lambdas[n]])
        a_dist = get_action_dist(get_combined_U(U_S, U_L, rho, C_social, actions), actions, alpha=alpha)
        a_probs_list.append([a_dist[a] for a in actions])
        for j, a in enumerate(actions):
            mse_list.append(np.square(a_dist[a] - a_prob_mean_all[n][j]))


    lambdas_extended = [extended_C_mean_all[n][-1]/L0_noncooperative_intent_given_action_probs[-1] for n in range(len(extended_C_mean_all))]
    for n in range(len(lambdas_extended)):
        C_social = make_dict(actions, [L0_noncooperative_intent_given_action_probs[0]*lambdas_extended[n], 
                                    L0_noncooperative_intent_given_action_probs[1]*lambdas_extended[n], 
                                    L0_noncooperative_intent_given_action_probs[2]*lambdas_extended[n]])
        a_dist = get_action_dist(get_combined_U(U_S, U_L, rho, C_social, actions), actions, alpha=alpha)
        a_probs_list.append([a_dist[a] for a in actions])
        for j, a in enumerate(actions):
            mse_list.append(np.square(a_dist[a] - extended_a_prob_mean_all[n][j]))

    return np.mean(mse_list)


def plot_social_cost_pattern_robustness_analysis(savepath=None):
    a_colors = ['#66A182', 'orange', "red"]

    y, x = np.meshgrid(np.linspace(0.01, 0.99, 100), np.linspace(0.5, 0.99, 100))
    z = np.zeros((len(x), len(y)))

    mean_costs = np.mean(list(C_mean_all) + list(extended_C_mean_all), axis=0)

    for i in range(len(x)):
        for j in range(len(y)):
            meaning_prob = x[i, j]
            prosocial_prior_prob = y[i, j]
            data_mse = cost_pattern_MSE(meaning_prob, prosocial_prior_prob, mean_costs)
            z[i, j] = data_mse

    z_min, z_max = z.min(), z.max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='gist_gray_r', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax, label='MSE')

    c = ax.contour(x, y, z, levels=[0, 0.001, 0.005, 0.01], vmin=z_min, vmax=z_max)
    ax.clabel(c, fontsize=9, inline=True, fmt='%.3f')

    ax.axis([x.min(), x.max(), y.min(), y.max()])
    plt.xlabel('Likelihood of intended meaning')
    plt.ylabel('Prior over '+r'$L_0$'+'\'s cooperative intent')

    inset_fig_params_all = [
        {'axes_params':[0.22, 0.2, 0.1, 0.1], 'meaning_certainty':0.7, 'prosocial_prior_prob':0.15, 'xytext':[0.67, 0.2], 'marker':'^'},
        {'axes_params':[0.42, 0.76, 0.1, 0.1], 'meaning_certainty':0.9, 'prosocial_prior_prob':0.85, 'xytext':[0.82, 0.88], 'marker':'*'},
        {'axes_params':[0.6, 0.42, 0.1, 0.1], 'meaning_certainty':0.85, 'prosocial_prior_prob':0.65, 'xytext':[0.9, 0.55], 'marker':'o'},
    ]

    for inset_fig_params in inset_fig_params_all:
        meaning_certainty = inset_fig_params['meaning_certainty']
        prosocial_prior_prob = inset_fig_params['prosocial_prior_prob']
        ax2 = fig.add_axes(inset_fig_params['axes_params'])
        for idx, action in enumerate(actions):
            ax2.bar(idx, infer_noncooperative_intent_of_L0(action, meaning_certainty, prosocial_prior_prob), 
                edgecolor=a_colors[idx],  facecolor='w', hatch='//////', width=0.65)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.patch.set_alpha(0)
        ax.plot(meaning_certainty, prosocial_prior_prob, inset_fig_params['marker'], c='indigo')
        ax.annotate("",
                    xy=(meaning_certainty, prosocial_prior_prob), xycoords='data',
                    xytext=inset_fig_params['xytext'], textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    custom_lines = [Line2D([0], [0], color='indigo', ls='none', marker='*'),
                    Line2D([0], [0], color='indigo', ls='none', marker='o'),
                    Line2D([0], [0], color='indigo', ls='none', marker='^')]

    ax.legend(custom_lines, ['Parameter combination fitted on data from Bridgers et al. (2023)', 
                            'Alternative combination that produces similar qualitative pattern', 
                            'Other combination in the parameter space'], bbox_to_anchor=(0.5, 1.02), loc='lower center', )

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()



def load_mean_understanding_ratings():
    path = 'data/understanding_data.csv'
    df = pd.read_csv(path)

    condition_dict = json.load(open('data/stimuli_info/loophole_prediction_condition_dict.json', 'r'))
    stories = condition_dict['stories']
    behaviors = ['Comply', 'Loophole', 'NonComply']
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
        
        response_lower = response.lower()

        if response_lower.endswith('funny') or response_lower.endswith('trouble'):
            continue

        cut = 2 if response.startswith('Yes') or response.startswith('No') else 1
        response = ' '.join(response.split()[cut:])
        
        # Normalize typo or stylistic variant in the wording of the options
        if response == 'may have undertstood':
            response = 'may have understood'
        elif response == 'did not understand at all':
            response = 'didn\'t understand at all'
        data[story][behavior].append(option2score[response])

    mean_understanding_ratings = [np.mean([np.mean(data[story][behavior]) for story in stories]) for behavior in behaviors]
    return mean_understanding_ratings


def plot_understanding_pattern_robustness_analysis(savepath=None):
    a_colors = ['#66A182', 'orange', "red"]

    mean_understanding_ratings = load_mean_understanding_ratings()
    human_normalized_understanding_ratings = [rating/3 for rating in mean_understanding_ratings]

    # print(mean_understanding_ratings) #[2.913, 2.334, 2.568]

    y, x = np.meshgrid(np.linspace(0.01, 0.99, 100), np.linspace(0.5, 0.99, 100))
    z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            meaning_prob = x[i, j]
            prosocial_prior_prob = y[i, j]
            data_mse = understanding_pattern_MSE(meaning_prob, prosocial_prior_prob, human_normalized_understanding_ratings)
            z[i, j] = data_mse

    z_min, z_max = z.min(), z.max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='gist_gray_r', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax,  label='MSE')

    c = ax.contour(x, y, z, levels=[0, 0.0001, 0.001, 0.005, 0.01], vmin=z_min, vmax=z_max)
    ax.clabel(c, fontsize=9, inline=True, fmt='%.3f')

    ax.axis([x.min(), x.max(), y.min(), y.max()])
    plt.xlabel('Likelihood of intended meaning')
    plt.ylabel('Prior over '+r'$L_0$'+'\'s cooperative intent')

    inset_fig_params_all = [
        {'axes_params':[0.18, 0.63, 0.1, 0.1], 'meaning_certainty':0.6, 'prosocial_prior_prob':0.9, 'xytext':[0.6, 0.82], 'marker':'^'},
        {'axes_params':[0.42, 0.76, 0.1, 0.1], 'meaning_certainty':0.9, 'prosocial_prior_prob':0.85, 'xytext':[0.82, 0.88], 'marker':'*'},
        {'axes_params':[0.6, 0.42, 0.1, 0.1], 'meaning_certainty':0.85, 'prosocial_prior_prob':0.75, 'xytext':[0.9, 0.55], 'marker':'o'},
    ]

    for inset_fig_params in inset_fig_params_all:
        meaning_certainty = inset_fig_params['meaning_certainty']
        prosocial_prior_prob = inset_fig_params['prosocial_prior_prob']
        ax2 = fig.add_axes(inset_fig_params['axes_params'])
        for idx, action in enumerate(actions):
            ax2.bar(idx, infer_understanding_of_L0(action, meaning_certainty, prosocial_prior_prob), 
                edgecolor=a_colors[idx],  facecolor='w', hatch='//////', width=0.65)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.patch.set_alpha(0)
        ax.plot(meaning_certainty, prosocial_prior_prob, inset_fig_params['marker'], c='indigo')
        ax.annotate("",
                    xy=(meaning_certainty, prosocial_prior_prob), xycoords='data',
                    xytext=inset_fig_params['xytext'], textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    custom_lines = [Line2D([0], [0], color='indigo', ls='none', marker='*'),
                    Line2D([0], [0], color='indigo', ls='none', marker='o'),
                    Line2D([0], [0], color='indigo', ls='none', marker='^')]

    ax.legend(custom_lines, ['Parameter combination fitted on data from Bridgers et al. (2023)', 
                            'Alternative combination that produces similar qualitative pattern', 
                            'Other combination in the parameter space'], bbox_to_anchor=(0.5, 1.02), loc='lower center', )

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def plot_a_prob_pattern_robusteness_analysis_for_rho_alpha(savepath=None):
    a_colors = ['#66A182', 'orange', "red"]

    # generate 2 2d grids for the x & y bounds
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(0, 2, 100))

    z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            rho = x[i, j]
            alpha = y[i, j]
            data_mse = action_prob_pattern_MSE(0.9, 0.85, rho, alpha)
            z[i, j] = data_mse

    z_min, z_max = z.min(), z.max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='gist_gray_r', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax,  label='MSE')

    c = ax.contour(x, y, z, levels=[0,0.00001, 0.0005, 0.003, 0.004, 0.006, 0.008, 0.01], vmin=z_min, vmax=z_max)
    ax.clabel(c, fontsize=9, inline=True, fmt='%.3f') 

    ax.axis([x.min(), x.max(), y.min(), y.max()])

    plt.xlabel('Utility trade-off '+r'$\rho$')
    plt.ylabel('Softmax optimality '+r'$\alpha$')

    U_S_intended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, -1, -1])
    U_S_unintended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, 1, -1])

    U_S = make_dict(actions, np.array([U_S_intended[action]*0.9 + U_S_unintended[action]*0.1 for action in actions]))
    U_L = make_dict(actions, np.array([-1, 0.7, 1])) 

    P_noncooperative_intents = [infer_noncooperative_intent_of_L0(action, 0.9, 0.85) for action in actions]

    inset_fig_params_all = [
        {'axes_params':[0.63, 0.74, 0.1, 0.1], 'rho':0.9, 'alpha':1.4, 'xytext':[0.8, 1.6], 'marker':'^'},
        {'axes_params':[0.36, 0.74, 0.1, 0.1], 'rho':0.34, 'alpha':1.56, 'xytext':[0, 1.6], 'marker':'*'},
        {'axes_params':[0.2, 0.24, 0.1, 0.1], 'rho':-0.6, 'alpha':0.1, 'xytext':[-0.6, 0.25], 'marker':'^'},
        {'axes_params':[0.32, 0.42, 0.1, 0.1], 'rho':0.4, 'alpha':1, 'xytext':[0, 0.9], 'marker':'o'},
    ]

    for inset_fig_params in inset_fig_params_all:
        ax2 = fig.add_axes(inset_fig_params['axes_params'])
        rho = inset_fig_params['rho']
        alpha = inset_fig_params['alpha']

        a_probs_list = []

        scales = np.arange(0, 6, 0.1)
        for scale in scales:
            C_social = make_dict(actions, [P_noncooperative_intents[0]*scale, 
                                        P_noncooperative_intents[1]*scale, 
                                        P_noncooperative_intents[2]*scale])
            a_dist = get_action_dist(get_combined_U(U_S, U_L, rho, C_social, actions), actions, alpha=alpha)
            a_probs_list.append([a_dist[a] for a in actions])

        for i, action in enumerate(actions):
            ax2.plot(scales*P_noncooperative_intents[2],  [a_probs[i] for a_probs in a_probs_list], '-', color=a_colors[i])

        ax2.set_ylim(0, 0.7)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.patch.set_alpha(0)
        ax.plot(rho, alpha, inset_fig_params['marker'], c='indigo')
        ax.annotate("",
                    xy=(rho, alpha), xycoords='data',
                    xytext=inset_fig_params['xytext'], textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))


    custom_lines = [Line2D([0], [0], color='indigo', ls='none', marker='*'),
                    Line2D([0], [0], color='indigo', ls='none', marker='o'),
                    Line2D([0], [0], color='indigo', ls='none', marker='^')]

    ax.legend(custom_lines, ['Parameter combination fitted on data from Bridgers et al. (2023)', 
                            'Alternative combination that produces similar qualitative pattern', 
                            'Other combination in the parameter space'], bbox_to_anchor=(0.5, 1.02), loc='lower center', )

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()


def plot_a_prob_pattern_robusteness_analysis_for_meaning_certainty_prosocial_prob(savepath=None):
    a_colors = ['#66A182', 'orange', "red"]

    x, y = np.meshgrid(np.linspace(0.5, 0.99, 100), np.linspace(0.01, 0.99, 100))

    z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            meaning_prob = x[i, j]
            prosocial_prior_prob = y[i, j]
            data_mse = action_prob_pattern_MSE(meaning_prob, prosocial_prior_prob, 0.3, 1.5)
            z[i, j] = data_mse

    z_min, z_max = z.min(), z.max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='gist_gray_r', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax, label='MSE')

    c = ax.contour(x, y, z, levels=[0.00001, 0.0005, 0.001, 0.002, 0.003, 0.0035, 0.004, 0.006, 0.008, 0.01], vmin=z_min, vmax=z_max)
    ax.clabel(c, fontsize=9, inline=True, fmt='%.3f')

    ax.axis([x.min(), x.max(), y.min(), y.max()])
    plt.xlabel('Likelihood of intended meaning')
    plt.ylabel('Prior over '+r'$L_0$'+'\'s cooperative intent')

    inset_fig_params_all = [
        {'axes_params':[0.25, 0.55, 0.1, 0.1], 'meaning_certainty':0.75, 'prosocial_prior_prob':0.5, 'xytext':[0.68, 0.6], 'marker':'o'},
        {'axes_params':[0.4, 0.75, 0.1, 0.1], 'meaning_certainty':0.9, 'prosocial_prior_prob':0.85, 'xytext':[0.8, 0.92], 'marker':'*'},
        {'axes_params':[0.5, 0.15, 0.1, 0.1], 'meaning_certainty':0.95, 'prosocial_prior_prob':0.25, 'xytext':[0.88, 0.15], 'marker':'^'},
        {'axes_params':[0.2, 0.25, 0.1, 0.1], 'meaning_certainty':0.6, 'prosocial_prior_prob':0.05, 'xytext':[0.6, 0.15], 'marker':'^'},
    ]

    for inset_fig_params in inset_fig_params_all:
        ax2 = fig.add_axes(inset_fig_params['axes_params'])
        meaning_certainty = inset_fig_params['meaning_certainty']
        prosocial_prior_prob= inset_fig_params['prosocial_prior_prob']
        rho = 0.34
        alpha = 1.56

        U_S_intended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, -1, -1])
        U_S_unintended = make_dict(['compliance', 'loophole', 'noncompliance'], [1, 1, -1])

        U_S = make_dict(actions, np.array([U_S_intended[action]*meaning_certainty + U_S_unintended[action]*(1 - meaning_certainty) for action in actions]))
        U_L = make_dict(actions, np.array([-1, 0.7, 1])) 

        P_noncooperative_intents = [infer_noncooperative_intent_of_L0(action, meaning_certainty, prosocial_prior_prob) for action in actions]


        a_probs_list = []

        scales = np.arange(0, 6, 0.1)
        for scale in scales:
            C_social = make_dict(actions, [P_noncooperative_intents[0]*scale, 
                                        P_noncooperative_intents[1]*scale, 
                                        P_noncooperative_intents[2]*scale])
            a_dist = get_action_dist(get_combined_U(U_S, U_L, rho, C_social, actions), actions, alpha=alpha)
            a_probs_list.append([a_dist[a] for a in actions])

        for i, action in enumerate(actions):
            ax2.plot(scales*P_noncooperative_intents[2],  [a_probs[i] for a_probs in a_probs_list], '-', color=a_colors[i])

        ax2.set_ylim(0, 1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.patch.set_alpha(0)
        ax.plot(meaning_certainty, prosocial_prior_prob, inset_fig_params['marker'], c='indigo')
        ax.annotate("",
                    xy=(meaning_certainty, prosocial_prior_prob), xycoords='data',
                    xytext=inset_fig_params['xytext'], textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    custom_lines = [Line2D([0], [0], color='indigo', ls='none', marker='*'),
                    Line2D([0], [0], color='indigo', ls='none', marker='o'),
                    Line2D([0], [0], color='indigo', ls='none', marker='^')]

    ax.legend(custom_lines, ['Parameter combination fitted on data from Bridgers et al. (2023)', 
                            'Alternative combination that produces similar qualitative pattern', 
                            'Other combination in the parameter space'], bbox_to_anchor=(0.5, 1.02), loc='lower center', )

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    # plt.savefig('fig/model_robustness_meaning_certainty_and_prosocial_prior_with_inset_plots.png', bbox_inches='tight')
    plt.show()



plot_social_cost_pattern_robustness_analysis(savepath='fig/model_robustness_cost_diff_mse_contour.pdf')

plot_understanding_pattern_robustness_analysis(savepath='fig/model_robustness_understanding_diff_mse_contour.pdf')

plot_a_prob_pattern_robusteness_analysis_for_rho_alpha(savepath='fig/model_robustness_contour_with_inset_plots.pdf')

plot_a_prob_pattern_robusteness_analysis_for_meaning_certainty_prosocial_prob(savepath='fig/model_robustness_meaning_certainty_and_prosocial_prior_with_inset_plots.pdf')

