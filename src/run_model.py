from model import *
import json
import numpy as np
import pandas as pd
import scipy.optimize


actions = ['compliance', 'loophole', 'noncompliance']

print('\nP(gamma | action)\n[Derived posterior of non-cooperative social intent given behavior]')
print('-'*27)
for action in actions:
    print('{:<20} {:.3f}'.format(action, infer_noncooperative_intent_of_L0(action, meaning_certainty=0.9, prosocial_prior_prob=0.85)))


print('\nP(intended meaning | action)\n[Derived posterior of understanding given behavior]')
print('-'*27)
for action in actions:
    print('{:<20} {:.3f}'.format(action, infer_understanding_of_L0(action, meaning_certainty=0.9, prosocial_prior_prob=0.85)))


print('\nDistinctiveness(action)\n[Derived measure of disctinve support for the ambivalence in the listener\'s social intent]')
print('-'*27)
for action in actions:
    print('{:<20} {:.3f}'.format(action, calc_strength_of_humor_as_distinctiveness_support(action, meaning_certainty=0.9, prosocial_prior_prob=0.85)))


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


def calc_a_pred_model_human_mse(params):
    rho, alpha = params

    actions = ['compliance', 'loophole', 'noncompliance']
    U_S = make_dict(actions, np.array([1, -0.8, -1]))
    U_L = make_dict(actions, np.array([-1, 0.7, 1])) 
    noncooperative_intent_given_action_probs = [infer_noncooperative_intent_of_L0(action, meaning_certainty=0.9, prosocial_prior_prob=0.85) for action in actions]

    lambdas = []
    for n, cost_of_nonobedience in enumerate(cost_of_nonobedience_all):
        lambdas.append(cost_of_nonobedience/noncooperative_intent_given_action_probs[-1])  # C_social = lambda*P_noncooperative; lambda = C_social/P_noncooperative
    
    mse_list = []
    a_probs_list = []
    
    for n, cost_of_nonobedience in enumerate(cost_of_nonobedience_all):
        C_social = make_dict(actions, [noncooperative_intent_given_action_probs[k]*lambdas[n] for k, _ in enumerate(actions)])
        a_dist = get_action_dist(get_combined_U(U_S, U_L, rho, C_social, actions), actions, alpha=alpha)
        a_probs_list.append([a_dist[a] for a in actions])
        for j, a in enumerate(actions):
            mse_list.append(np.square(a_dist[a] - a_prob_mean_all[n][j]))
    return np.mean(mse_list)


params_fit = scipy.optimize.minimize(calc_a_pred_model_human_mse, [0, 1])
# print(rs)
# print()

rho_fit, alpha_fit = params_fit.x[0], params_fit.x[1]

for n, cost_of_nonobedience in enumerate(cost_of_nonobedience_all):
    print(cost_of_nonobedience/infer_noncooperative_intent_of_L0('noncompliance', 0.9, 0.85)) 

print('\nEstimated lambdas (scale of punishment) for different power relation conditons: ')
for n, cost_of_nonobedience in enumerate(cost_of_nonobedience_all):
    print('lambda_{} = {:.3f}'.format(power_relations[n].upper(), cost_of_nonobedience/infer_noncooperative_intent_of_L0('noncompliance', 0.9, 0.85))) 

print('\nHyperparameters optimized based on action prediction data from Brigders et al:\nrho={:.3f}, alpha={:.3f}'.format(rho_fit, alpha_fit))


