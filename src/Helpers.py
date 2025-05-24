from Chord import Chord
from Rule import Rule
import numpy as np
def get_possible_intervals(chord1,chord2):

    intervals = []
    intervals.append([str(chord1.distance_to(chord1)), str(chord1.distance_to(chord2))])
    intervals.append([str(chord1.distance_from(chord2)), str(chord2.distance_to(chord2))])

    return intervals

def get_qualities(chord1,chord2):
    return [chord1.quality, chord2.quality]

def get_possible_rhs_from_str(chord1,chord2):
    return [(get_possible_intervals(chord1, chord2)[0], get_qualities(chord1, chord2)), (get_possible_intervals(chord1, chord2)[1], get_qualities(chord1, chord2))]

def get_applicable_rules(rhs_list, rules):
    applicable_rules = []
    for rule in rules:
        for rhs in rhs_list:
            if rule.rhs() == rhs:
                applicable_rules.append(rule)
    return applicable_rules

def evaluate_tree_from_dict(tree, prob_dict,parent_label=None):
    applied_rules = []
    log_probs = 0
    if tree.get('children'):
        children = tree['children']
        
        if len(children) == 2:
            chord1 = Chord(children[0]['label'])
            chord2 = Chord(children[1]['label'])
            parent = Chord(tree['label'])
            parent_quality = parent.quality

            if chord1.distance_to(parent) == 0:
                chord1interval = 0
                chord2interval = chord1.distance_to(chord2)
            elif chord2.distance_to(parent) == 0:
                chord1interval = chord2.distance_to(chord1)
                chord2interval = 0
            else:
                # If neither child matches the parent, use distance from parent
                chord1interval = chord1.distance_to(parent)
                chord2interval = chord2.distance_to(parent)

            child_intervals = [str(chord1interval), str(chord2interval)]
            child_qualities = [chord1.quality, chord2.quality]

            rule = Rule(parent_quality, child_intervals, child_qualities)
            applied_rules.append(rule)

        # Recursively gather rules from children
        for child in children:
            applied_rules.extend(evaluate_tree_from_dict(child, prob_dict,tree['label'])[0])

    for rule in applied_rules:
        rule_hash = rule.make_hashable()
        if rule_hash in prob_dict:
            log_probs += np.log(prob_dict[rule_hash])
        else:
            log_probs += np.log(1e-10)
    return applied_rules, log_probs
