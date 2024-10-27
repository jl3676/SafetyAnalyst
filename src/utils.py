import json
from json_repair import repair_json
from src.models import SafetyReporter
import ray

def check_valid_harm_benefit_tree_helper(harm_benefit_tree, 
                                         analysis_type=None):
    if harm_benefit_tree is None or harm_benefit_tree == []:
        return False
    try:
        for stakeholder_entry in harm_benefit_tree:
            _ = stakeholder_entry['stakeholder']
            harms = [] if 'benefit' in analysis_type else stakeholder_entry['harms']
            benefits = [] if 'harm' in analysis_type else stakeholder_entry['benefits']
            for harm in harms:
                effects = harm['effects']
                for effect in effects:
                    _ = effect['extent']
                    _ = effect['likelihood']
                    _ = effect['immediacy']
            for benefit in benefits:
                effects = benefit['effects']
                for effect in effects:
                    _ = effect['extent']
                    _ = effect['likelihood']
                    _ = effect['immediacy']
    except:
        return False
    return True


def valid_tree(string: str,
               analysis_type=None):
    try:
        harm_benefit_tree = "]".join(("[" + "[".join(string.split("[")[1:])).split("]")[:-1]) + "]"
        harm_benefit_tree = json.loads(repair_json(harm_benefit_tree))
        if not check_valid_harm_benefit_tree_helper(harm_benefit_tree, analysis_type):
            raise ValueError()
        return True
    except:
        return False
    

def generate_trees(prompts, 
                   safety_reporter: SafetyReporter, 
                   analysis_type: str = None,
                   trees: list[str] = None,
                   successful: list[bool] = None):
    trees = trees if trees else [''] * len(prompts)
    successful = successful if successful else [False] * len(prompts)
    remaining_prompts = [prompt for i, prompt in enumerate(prompts) if not successful[i]]
    outputs = safety_reporter.batched_generate.remote(prompts=remaining_prompts)
    remaining_trees = ray.get(outputs)
    trees = [tree if successful[i] else remaining_trees.pop(0) for i, tree in enumerate(trees)]
    for i, tree in enumerate(trees):
        if not successful[i] and valid_tree(tree, analysis_type):
            successful[i] = True

    return trees, successful

def return_trees_JSON(trees):
    trees_json = []
    for tree in trees:
        tree = "]".join(("[" + "[".join(tree.split("[")[1:])).split("]")[:-1]) + "]"
        try:
            tree = json.loads(repair_json(tree))
        except:
            tree = []
        trees_json.append(tree)
    return trees_json

def combine_trees(harm_trees: list[str], benefit_trees: list[str]):
    harm_benefit_trees = []
    for harm_tree, benefit_tree in zip(harm_trees, benefit_trees):
        harm_benefit_tree = harm_tree + benefit_tree
        harm_benefit_trees.append(harm_benefit_tree)
    return harm_benefit_trees


def pretty_print(prompt, harm_benefit_tree, analysis_type='both'):
    print(f"PROMPT: {prompt}\n")
    for stakeholder_entry in harm_benefit_tree:
        stakeholder = stakeholder_entry['stakeholder']
        print(f"  Stakeholder: {stakeholder}")

        harms = stakeholder_entry['harms'] if 'benefit' not in analysis_type else []
        benefits = stakeholder_entry['benefits'] if 'harm' not in analysis_type else []

        print("    Harms:")
        for harm in harms:
            action = harm['action']
            print(f"      Action: {action}")

            effects = harm['effects']
            for effect in effects:
                try:
                    extent = effect['extent']
                    likelihood = effect['likelihood']
                    immediacy = effect['immediacy']
                    print(f"        Effect: {effect['effect']}")
                    print(f"          Extent: {extent}")
                    print(f"          Likelihood: {likelihood}")
                    print(f"          Immediate: {immediacy}")
                except:
                    continue
        
        print("    Benefits:")
        for benefit in benefits:
            action = benefit['action']
            print(f"      Action: {action}")

            effects = benefit['effects']
            for effect in effects:
                try:
                    extent = effect['extent']
                    likelihood = effect['likelihood']
                    immediacy = effect['immediacy']
                    print(f"        Effect: {effect['effect']}")
                    print(f"          Extent: {extent}")
                    print(f"          Likelihood: {likelihood}")
                    print(f"        Immediate: {immediacy}")
                except:
                    continue

    print("\n")