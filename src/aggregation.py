import argparse
import numpy as np
import json
from datasets import load_dataset
from scipy.optimize import differential_evolution


def arg_parser():
    parser = argparse.ArgumentParser(description='Aggregate harms and benefits')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file in .jsonl format containing "prompt", "label", and "harm_benefit_tree" fields')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model that generated the input file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Input dataset name')
    parser.add_argument('--analysis_type', type=str, required=True, help='Type of analysis: harms or benefits or both')
    parser.add_argument('--use_action_weights', action='store_true', help='Include action weights')
    parser.add_argument('--use_effect_weights', action='store_true', help='Include effect weights')
    parser.add_argument('--fit', action='store_true', help='Fit the model to input data')
    return parser.parse_args()


def parse_extent(extent, param=None):
    if type(extent) == str:
        if extent.lower() == 'minor':
            return param[0]
        elif extent.lower() == 'significant':
            return param[1]
        elif extent.lower() == 'substantial':
            return param[2]
        elif extent.lower() == 'major':
            return param[3]
    return 0.0
    

def parse_likelihood(likelihood, param=None):
    if type(likelihood) == str:
        if likelihood.lower() == 'low':
            return param[0]
        elif likelihood.lower() == 'medium':
            return param[1]
        elif likelihood.lower() == 'high':
            return param[2]
    return 0.0


def get_action_weight(action, category_weights):
    category_names = ['Security Risks', 'Operational Misuses', 'Violence & Extremism', 'Hate/Toxicity', 'Sexual Content', 'Child Harm', 'Self-harm','Political Usage', 'Economic Harm', 'Deception', 'Manipulation', 'Defamation', 'Fundamental Rights', 'Discrimination/Bias', 'Privacy', 'Criminal Activities']
    for c, category in enumerate(category_names):
        if category in action:
            return category_weights[c]
    return 0


def get_effect_weight(effect, category_weights):
    category_names = ['Physical harm', 'Psychological', 'Social', 'Property', 'Liberty', 'Collective', 'Ecological']
    for c, category in enumerate(category_names):
        if category in effect:
            return category_weights[c]
    return 0


def unpack_params(params, use_action_weights, use_effect_weights):
    if use_action_weights:
        category_weights_actions = params[:16]
        extent_minor_weight_harms, extent_significant_weight_harms, extent_substantial_weight_harms = params[16:19]
        likelihood_low_weight_harms, likelihood_medium_weight_harms = params[19:21]
        discount_rate_harms = params[21]
        extent_minor_weight_benefits, extent_significant_weight_benefits, extent_substantial_weight_benefits = params[22:25]
        likelihood_low_weight_benefits, likelihood_medium_weight_benefits = params[25:27]
        discount_rate_benefits = params[27]
        benefit_discount_rate = params[28]
        if use_effect_weights:
            category_weights_effects = params[29:36]
        else:
            category_weights_effects = [1] * 7
    elif use_effect_weights:
        category_weights_effects = params[:7]
        category_weights_actions = [1] * 16
        extent_minor_weight_harms, extent_significant_weight_harms, extent_substantial_weight_harms = params[7:10]
        likelihood_low_weight_harms, likelihood_medium_weight_harms = params[10:12]
        discount_rate_harms = params[12]
        extent_minor_weight_benefits, extent_significant_weight_benefits, extent_substantial_weight_benefits = params[13:16]
        likelihood_low_weight_benefits, likelihood_medium_weight_benefits = params[16:18]
        discount_rate_benefits = params[18]
        benefit_discount_rate = params[19]

    extent_major_weight_harms = 1
    extent_substantial_weight_harms *= extent_major_weight_harms
    extent_significant_weight_harms *= extent_substantial_weight_harms
    extent_minor_weight_harms *= extent_significant_weight_harms

    extent_major_weight_benefits = 1
    extent_substantial_weight_benefits *= extent_major_weight_benefits
    extent_significant_weight_benefits *= extent_substantial_weight_benefits
    extent_minor_weight_benefits *= extent_significant_weight_benefits

    likelihood_high_weight_harms = 1
    likelihood_medium_weight_harms *= likelihood_high_weight_harms
    likelihood_low_weight_harms *= likelihood_medium_weight_harms

    likelihood_high_weight_benefits = 1
    likelihood_medium_weight_benefits *= likelihood_high_weight_benefits
    likelihood_low_weight_benefits *= likelihood_medium_weight_benefits

    return extent_minor_weight_harms, extent_significant_weight_harms, extent_substantial_weight_harms, extent_major_weight_harms, likelihood_low_weight_harms, likelihood_medium_weight_harms, likelihood_high_weight_harms, discount_rate_harms, extent_minor_weight_benefits, extent_significant_weight_benefits, extent_substantial_weight_benefits, extent_major_weight_benefits, likelihood_low_weight_benefits, likelihood_medium_weight_benefits, likelihood_high_weight_benefits, discount_rate_benefits, benefit_discount_rate, category_weights_actions, category_weights_effects


def get_action_validity(action, valid_actions):
    if valid_actions is None:
        return True
    for a in valid_actions:
        if a in action:
            return True
    return False


def mathematical_model_nllh(params, 
                            harm_benefit_trees, 
                            labels, 
                            analysis_type=None, 
                            valid_actions=None, 
                            use_action_weights=False, 
                            use_effect_weights=False):
    
    # unpack parameters
    extent_minor_weight_harms, extent_significant_weight_harms, extent_substantial_weight_harms, extent_major_weight_harms, likelihood_low_weight_harms, likelihood_medium_weight_harms, likelihood_high_weight_harms, discount_rate_harms, extent_minor_weight_benefits, extent_significant_weight_benefits, extent_substantial_weight_benefits, extent_major_weight_benefits, likelihood_low_weight_benefits, likelihood_medium_weight_benefits, likelihood_high_weight_benefits, discount_rate_benefits, benefit_discount_rate, category_weights_actions, category_weights_effects = unpack_params(params, use_action_weights, use_effect_weights)

    harmfulness_scores = []
    for line in harm_benefit_trees:
        line_json = json.loads(line)
        if 'messages' in line_json:
            harm_benefit_analysis = json.loads(line_json['messages'][-1]['content'])
        else:
            harm_benefit_analysis = json.loads(line_json['response'])
        harmfulness_score = 0
        for stakeholder_entry in harm_benefit_analysis:
            entries = []
            entries_type = []
            try:
                if analysis_type != 'benefits' and 'harms' in stakeholder_entry and stakeholder_entry['harms'] is not None:
                    entries.append(stakeholder_entry['harms'])
                    entries_type.append('harms')
            except:
                pass
            try:
                if analysis_type != 'harms' and 'benefits' in stakeholder_entry and stakeholder_entry['benefits'] is not None:
                    entries.append(stakeholder_entry['benefits'])
                    entries_type.append('benefits')
            except:
                pass

            for b, this_entries in enumerate(entries):
                extent_minor_weight = extent_minor_weight_harms if entries_type[b] == 'harms' else extent_minor_weight_benefits
                extent_significant_weight = extent_significant_weight_harms if entries_type[b] == 'harms' else extent_significant_weight_benefits
                extent_substantial_weight = extent_substantial_weight_harms if entries_type[b] == 'harms' else extent_substantial_weight_benefits
                extent_major_weight = extent_major_weight_harms if entries_type[b] == 'harms' else extent_major_weight_benefits
                likelihood_low_weight = likelihood_low_weight_harms if entries_type[b] == 'harms' else likelihood_low_weight_benefits
                likelihood_medium_weight = likelihood_medium_weight_harms if entries_type[b] == 'harms' else likelihood_medium_weight_benefits
                likelihood_high_weight = likelihood_high_weight_harms if entries_type[b] == 'harms' else likelihood_high_weight_benefits
                discount_rate = discount_rate_harms if entries_type[b] == 'harms' else discount_rate_benefits
                for action_entry in this_entries:
                    try:
                        action_weight = get_action_weight(action_entry['action'], category_weights_actions) if use_action_weights and entries_type[b] == 'harms' else 1
                        if entries_type[b] == 'harms':
                            if not get_action_validity(action_entry['action'], valid_actions):
                                continue
                        for effect in action_entry['effects']:
                            try:
                                effect_weight = get_effect_weight(effect['effect'], category_weights_effects) if use_effect_weights else 1
                                immediacy_inv = 1 - int(effect['immediacy'])
                                extent = parse_extent(effect['extent'], param=[extent_minor_weight, extent_significant_weight, extent_substantial_weight, extent_major_weight])
                                likelihood = parse_likelihood(effect['likelihood'], param=[likelihood_low_weight, likelihood_medium_weight, likelihood_high_weight])
                                harmfulness_score += action_weight * effect_weight * extent * likelihood * (discount_rate ** immediacy_inv) * (-benefit_discount_rate if entries_type[b] == 'benefits' else 1)
                            except:
                                pass
                    except:
                        pass
        harmfulness_scores.append(harmfulness_score)
    harmfulness_scores = np.array(harmfulness_scores)
    labels = np.array(labels)
    # transform harmfulness score to probability using sigmoid
    probs = 1 / (1 + np.exp(-harmfulness_scores))
    if analysis_type == 'harms':
        probs = (probs - 0.5) * 2
    elif analysis_type == 'benefits':
        probs *= 2 
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    # calculate negative log likelihood
    nllh = -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    return nllh


def mathematical_model(param, 
                       harm_benefit_trees, 
                       analysis_type=None, 
                       valid_actions=None,
                       use_action_weights=False,
                       use_effect_weights=False):
    # unpack parameters
    extent_minor_weight_harms, extent_significant_weight_harms, extent_substantial_weight_harms, extent_major_weight_harms, likelihood_low_weight_harms, likelihood_medium_weight_harms, likelihood_high_weight_harms, discount_rate_harms, extent_minor_weight_benefits, extent_significant_weight_benefits, extent_substantial_weight_benefits, extent_major_weight_benefits, likelihood_low_weight_benefits, likelihood_medium_weight_benefits, likelihood_high_weight_benefits, discount_rate_benefits, benefit_discount_rate, category_weights_actions, category_weights_effects = unpack_params(param, use_action_weights, use_effect_weights)

    harmfulness_scores = []
    for line in harm_benefit_trees:
        line_json = json.loads(line)
        if 'messages' in line_json:
            harm_benefit_analysis = json.loads(line_json['messages'][-1]['content'])
        elif 'response' in line_json:
            harm_benefit_analysis = json.loads(line_json['response'])
        harmfulness_score = 0
        for stakeholder_entry in harm_benefit_analysis:
            entries = []
            entries_type = []
            try:
                if analysis_type != 'benefits' and 'harms' in stakeholder_entry and stakeholder_entry['harms'] is not None:
                    entries.append(stakeholder_entry['harms'])
                    entries_type.append('harms')
            except:
                pass
            try:
                if analysis_type != 'harms' and 'benefits' in stakeholder_entry and stakeholder_entry['benefits'] is not None:
                    entries.append(stakeholder_entry['benefits'])
                    entries_type.append('benefits')
            except:
                pass

            for b, this_entries in enumerate(entries):
                extent_minor_weight = extent_minor_weight_harms if entries_type[b] == 'harms' else extent_minor_weight_benefits
                extent_significant_weight = extent_significant_weight_harms if entries_type[b] == 'harms' else extent_significant_weight_benefits
                extent_substantial_weight = extent_substantial_weight_harms if entries_type[b] == 'harms' else extent_substantial_weight_benefits
                extent_major_weight = extent_major_weight_harms if entries_type[b] == 'harms' else extent_major_weight_benefits
                likelihood_low_weight = likelihood_low_weight_harms if entries_type[b] == 'harms' else likelihood_low_weight_benefits
                likelihood_medium_weight = likelihood_medium_weight_harms if entries_type[b] == 'harms' else likelihood_medium_weight_benefits
                likelihood_high_weight = likelihood_high_weight_harms if entries_type[b] == 'harms' else likelihood_high_weight_benefits
                discount_rate = discount_rate_harms if entries_type[b] == 'harms' else discount_rate_benefits
                for action_entry in this_entries:
                    try:
                        action_weight = get_action_weight(action_entry['action'], category_weights_actions) if use_action_weights and entries_type[b] == 'harms' else 1
                        if entries_type[b] == 'harms':
                            if not get_action_validity(action_entry['action'], valid_actions):
                                continue
                        for effect in action_entry['effects']:
                            try:
                                effect_weight = get_effect_weight(effect['effect'], category_weights_effects) if use_effect_weights else 1
                                immediacy_inv = 1 - int(effect['immediacy'])
                                extent = parse_extent(effect['extent'], param=[extent_minor_weight, extent_significant_weight, extent_substantial_weight, extent_major_weight])
                                likelihood = parse_likelihood(effect['likelihood'], param=[likelihood_low_weight, likelihood_medium_weight, likelihood_high_weight])
                                harmfulness_score += action_weight * effect_weight * extent * likelihood * (discount_rate ** immediacy_inv) * (-benefit_discount_rate if entries_type[b] == 'benefits' else 1)
                            except:
                                pass
                    except:
                        pass
        harmfulness_scores.append(harmfulness_score)

    return np.array(harmfulness_scores)


def fit_model(args, data, labels, save_path, valid_actions=None):
    n_params = 36 # highest possible number of parameters
    bounds = [(0, 1)] * n_params
    result = differential_evolution(mathematical_model_nllh, 
                                    bounds=bounds, 
                                    args=(data, 
                                          labels, 
                                          args.analysis_type, 
                                          valid_actions,
                                          args.use_action_weights,
                                          args.use_effect_weights))
    np.save(save_path, result.x)
    return result.x


if __name__ == '__main__':
    args = arg_parser()
    data = load_dataset('json', data_files=args.input_path)['train']
    prompts = data['prompt']
    labels = data['label']
    harm_benefit_trees = data['harm_benefit_tree']
    suffix = f'_{args.dataset_name}'
    suffix += '_actions' if args.use_action_weights else ''
    suffix += '_effects' if args.use_effect_weights else ''
    param_path = f'saved_params/{args.model_name}_{args.analysis_type}{suffix}.npy'
    # fit model
    if args.fit:
        params = fit_model(args, harm_benefit_trees, labels, param_path)
    # load fitted parameters
    params = np.load(param_path)
    # get predictions
    harmfulness_scores = mathematical_model(params, 
                                            harm_benefit_trees,
                                            args.analysis_type,
                                            use_action_weights=args.use_action_weights,
                                            use_effect_weights=args.use_effect_weights)
    preds = (harmfulness_scores > 0).astype(int)
    # save predictions
    with open(args.output_path, 'w') as f:
        for pred in preds:
            f.write(f'{pred}\n')