import numpy as np
import scipy.special
from scipy.special import softmax
from scipy.special import kl_div


def normalize(arr):
    total = np.sum(arr)
    return [x/total for x in arr]


def get_action_dist(U, actions, alpha=1):
    utilities = [U[a]*alpha for a in actions]
    return dict(zip(actions, softmax(utilities)))


def get_combined_utility(U_S, U_L, rho, C_social, a):
    return rho*U_S[a] + U_L[a] - C_social[a]


def get_combined_U(U_S, U_L, rho, C_social, actions):
    return dict(zip(actions, [get_combined_utility(U_S, U_L, rho, C_social, a) for a in actions]))


def get_L0_U(U_S, gamma, actions):
    return dict(zip(actions, [gamma*U_S[a] for a in actions]))


def make_dict(actions, values):
    # A generic function that builds a dictionary for mapping an action to its value (utility or cost)
    return dict(zip(actions, values))


def infer_noncooperative_intent_of_L0(action, meaning_certainty, prosocial_prior_prob):
    actions = ['compliance', 'loophole', 'noncompliance']
    U_S_intended = make_dict(actions, [1, -1, -1])
    U_S_unintended = make_dict(actions, [1, 1, -1])

    gammas = [-1, 1]
    meanings = ['intended', 'unintended']

    gamma_probs = {-1:(1 - prosocial_prior_prob), 1:prosocial_prior_prob}
    meaning_probs = {'intended':meaning_certainty, 'unintended':(1 - meaning_certainty)}

    log_joint_prob_dict = {}

    for gamma in gammas:
        for meaning in meanings:
            U_S = U_S_intended if meaning == 'intended' else U_S_unintended
            action_lh_dict = get_action_dist(get_L0_U(U_S, gamma, actions), actions)
            log_joint_prob = np.log(action_lh_dict[action]) + np.log(gamma_probs[gamma]) + np.log(meaning_probs[meaning])
            log_joint_prob_dict['{}, {}'.format(meaning, gamma)] = log_joint_prob

    hypotheses = ['{}, {}'.format(meaning, gamma) for gamma in gammas for meaning in meanings]
    posteriors = scipy.special.softmax([log_joint_prob_dict[hyp] for hyp in hypotheses])
    posterior_prob_dict = dict(zip(hypotheses, posteriors))

    return posterior_prob_dict['intended, -1'] + posterior_prob_dict['unintended, -1']


def infer_understanding_of_L0(action, meaning_certainty, prosocial_prior_prob):
    actions = ['compliance', 'loophole', 'noncompliance']

    U_S_intended = make_dict(actions, [1, -1, -1])
    U_S_unintended = make_dict(actions, [1, 1, -1])

    gammas = [-1, 1]
    meanings = ['intended', 'unintended']

    gamma_probs = {-1:(1 - prosocial_prior_prob), 1:prosocial_prior_prob}
    meaning_probs = {'intended':meaning_certainty, 'unintended':(1 - meaning_certainty)}

    log_joint_prob_dict = {}

    for gamma in gammas:
        for meaning in meanings:
            U_S = U_S_intended if meaning == 'intended' else U_S_unintended
            action_lh_dict = get_action_dist(get_L0_U(U_S, gamma, actions), actions)
            log_joint_prob = np.log(action_lh_dict[action]) + np.log(gamma_probs[gamma]) + np.log(meaning_probs[meaning])
            log_joint_prob_dict['{}, {}'.format(meaning, gamma)] = log_joint_prob

    hypotheses = ['{}, {}'.format(meaning, gamma) for gamma in gammas for meaning in meanings]
    posteriors = scipy.special.softmax([log_joint_prob_dict[hyp] for hyp in hypotheses])
    posterior_prob_dict = dict(zip(hypotheses, posteriors))

    return posterior_prob_dict['intended, -1'] + posterior_prob_dict['intended, 1']


def get_meaning_dist_given_intent_and_action(action, gamma, meaning_certainty, prosocial_prior_prob):
    actions = ['compliance', 'loophole', 'noncompliance']

    U_S_intended = make_dict(actions, [1, -1, -1])
    U_S_unintended = make_dict(actions, [1, 1, -1])

    gammas = [-1, 1]
    meanings = ['intended', 'unintended']

    gamma_probs = {-1:(1 - prosocial_prior_prob), 1:prosocial_prior_prob}
    meaning_probs = {'intended':meaning_certainty, 'unintended':(1 - meaning_certainty)}

    log_joint_prob_dict = {}


    for meaning in meanings:
        U_S = U_S_intended if meaning == 'intended' else U_S_unintended
        action_lh_dict = get_action_dist(get_L0_U(U_S, gamma, actions), actions)
        log_joint_prob = np.log(action_lh_dict[action]) + np.log(gamma_probs[gamma]) + np.log(meaning_probs[meaning])
        log_joint_prob_dict['{}, {}'.format(meaning, gamma)] = log_joint_prob

    hypotheses = ['{}, {}'.format(meaning, gamma) for meaning in meanings]
    posteriors = scipy.special.softmax([log_joint_prob_dict[hyp] for hyp in hypotheses])
    posterior_prob_dict = dict(zip(meanings, posteriors))

    return posterior_prob_dict


def calc_strength_of_humor_as_distinctiveness_support(action, meaning_certainty, prosocial_prior_prob):
    meanings = ['intended', 'unintended']
    meaning_dist_given_cooperative_intent = get_meaning_dist_given_intent_and_action(action, 1, meaning_certainty, prosocial_prior_prob)
    meaning_dist_given_noncooperative_intent = get_meaning_dist_given_intent_and_action(action, -1, meaning_certainty, prosocial_prior_prob)
    meaning_probs_given_cooperative_intent = [meaning_dist_given_cooperative_intent[meaning] for meaning in meanings]
    meaning_probs_given_noncooperative_intent = [meaning_dist_given_noncooperative_intent[meaning] for meaning in meanings]
    return np.sum(kl_div(meaning_probs_given_cooperative_intent, meaning_probs_given_noncooperative_intent)) + np.sum(kl_div(meaning_probs_given_noncooperative_intent, meaning_probs_given_cooperative_intent))

