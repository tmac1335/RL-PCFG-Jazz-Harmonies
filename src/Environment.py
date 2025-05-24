import random
import numpy as np
from Chord import Chord
from TreeNode import TreeNode, create_parent_node, get_top_level_nodes
from Helpers import get_possible_intervals, get_qualities, get_possible_rhs_from_str, get_applicable_rules
from Rule import Rule
from Chord import Chord
from TreeNode import TreeNode, create_parent_node, get_top_level_nodes
import torch

class Environment:

    MAX_ACTIONS = 20
    MAX_CHORDS = 20

    def __init__(self, chord_sequence, rules, prob_dict):
        self.initial_sequence = chord_sequence
        self.current_state = chord_sequence.copy()
        self.current_nodes = [TreeNode(Chord(chord)) for chord in chord_sequence]
        self.rules = rules
        self.prob_dict = prob_dict
        self.actions = self.get_actions()
        self.applied_rules = []
        
    # Each index of rhs are a list of applicable rules to currentstate and currentstate + 1
    def get_actions(self):
        rhs_list = []

        for i in range(len(self.current_state) - 1):
            chord1 = Chord(self.current_state[i])
            chord2 = Chord(self.current_state[i + 1])
            possible_rhs = get_possible_rhs_from_str(chord1, chord2)
            rhs_list.append(get_applicable_rules(possible_rhs, self.rules))


        return rhs_list
    
    def concat_state_and_actions(self, chords_tensor, actions_tensor):
        # Concatenate along the last dimension
        return torch.cat((chords_tensor.reshape(-1), actions_tensor.squeeze(0).reshape(-1)), dim=-1)


    def build_action_index_map(self):

        index_map = []
        for i, rule_list in enumerate(self.get_actions()):
            for j, _ in enumerate(rule_list):
                index_map.append((i, j))
        return index_map

    def apply_rule_index(self,rule_index_i,rule_index_j):
        rule = self.actions[rule_index_i][rule_index_j]
        node1 = self.current_nodes[rule_index_i]
        node2 = self.current_nodes[rule_index_i + 1]
        nodes = [node1,node2]
        parent_root_index = rule.child_intervals.index('0') 
        parent = create_parent_node(node1, node2, nodes[parent_root_index], rule.lhs())
        self.current_nodes[rule_index_i] = parent
        self.current_nodes.pop(rule_index_i + 1)
        self.applied_rules.append(rule)
        self.current_nodes = get_top_level_nodes(self.current_nodes)
        self.current_state = [node.chord.label for node in self.current_nodes]
        self.actions = self.get_actions()

    def apply_rule(self, rule, node1, node2):
        nodes = [node1,node2]
        parent_root_index = rule.child_intervals.index('0') 
        parent = create_parent_node(node1, node2, nodes[parent_root_index], rule.lhs())

        self.current_nodes.append(parent)
        self.applied_rules.append(rule)

        self.current_nodes = get_top_level_nodes(self.current_nodes)
        self.current_state = [node.chord.label for node in self.current_nodes]
        self.actions = self.get_actions()

    def get_state_tensor2(self):
        QUALITY_VOCAB = ['major', 'minor', 'sus', 'unknown']
        INTERVAL_VOCAB = [str(i) for i in range(0,12)]

        def one_hot_encode(value, vocab):
            vec = [0] * len(vocab)
            idx = vocab.index(value if value in vocab else vocab.index('Other'))
            vec[idx] = 1
            return vec

        def encode_rule(rule):
            parent_vec = one_hot_encode(rule.parent_quality, QUALITY_VOCAB)

            interval1_vec = one_hot_encode(rule.child_intervals[0], INTERVAL_VOCAB)
            quality1_vec = one_hot_encode(rule.child_qualities[0], QUALITY_VOCAB)

            interval2_vec = one_hot_encode(rule.child_intervals[1], INTERVAL_VOCAB)
            quality2_vec = one_hot_encode(rule.child_qualities[1], QUALITY_VOCAB)

            return parent_vec + interval1_vec + quality1_vec + interval2_vec + quality2_vec

        chord_vector = []
        for chord in self.current_state[:self.MAX_CHORDS]:
            chord_vector.append(Chord.encode_chord(chord))

        # Pad if needed
        while len(chord_vector) < self.MAX_CHORDS:
            chord_vector.append([0] * len(chord_vector[0]))

        flat_chord_vector = [val for chord in chord_vector for val in chord]
        # chord_tensor = torch.tensor(flat_chord_vector, dtype=torch.float32).unsqueeze(0)

        actions = self.get_actions()
        action_vector = []

        count = 0
        for i in range(len(actions)):
            for j in range(len(actions[i])):
                if count >= self.MAX_ACTIONS:
                    break
                action_vector.append(encode_rule(actions[i][j]))
                count += 1
            if count >= self.MAX_ACTIONS:
                break
        
        flat_action_vector = [val for action in action_vector for val in action]

        return torch.tensor(flat_chord_vector + flat_action_vector, dtype=torch.float32)
        # rule_dim = len(action_vector[0]) if action_vector else 0
        # while len(action_vector) < self.MAX_ACTIONS:
        #     action_vector.append([0] * rule_dim)

        # actions_tensor = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0)
        # return chord_tensor, actions_tensor


    def get_state_tensor(self):
        QUALITY_VOCAB = ['major', 'minor', 'sus', 'unknown']
        INTERVAL_VOCAB = [str(i) for i in range(0,12)]

        def one_hot_encode(value, vocab):
            vec = [0] * len(vocab)
            idx = vocab.index(value if value in vocab else vocab.index('Other'))
            vec[idx] = 1
            return vec

        def encode_rule(rule):
            parent_vec = one_hot_encode(rule.parent_quality, QUALITY_VOCAB)

            interval1_vec = one_hot_encode(rule.child_intervals[0], INTERVAL_VOCAB)
            quality1_vec = one_hot_encode(rule.child_qualities[0], QUALITY_VOCAB)

            interval2_vec = one_hot_encode(rule.child_intervals[1], INTERVAL_VOCAB)
            quality2_vec = one_hot_encode(rule.child_qualities[1], QUALITY_VOCAB)

            return parent_vec + interval1_vec + quality1_vec + interval2_vec + quality2_vec

        chord_vector = []
        for chord in self.current_state[:self.MAX_CHORDS]:
            chord_vector.append(Chord.encode_chord(chord))
            distance_vec = []
            for other_chord in self.current_state[:self.MAX_CHORDS]:
                distance = Chord(chord).distance_to(Chord(other_chord))
                distance_vec.append(distance)
            while len(distance_vec) < self.MAX_CHORDS:
                distance_vec.append(-1)
            chord_vector[-1].extend(distance_vec)
        # Pad if needed
        while len(chord_vector) < self.MAX_CHORDS:
            chord_vector.append([0] * len(chord_vector[0]))

        flat_chord_vector = [val for chord in chord_vector for val in chord]
        chord_tensor = torch.tensor(flat_chord_vector, dtype=torch.float32).unsqueeze(0)

        actions = self.actions
        action_vector = []

        count = 0
        for i in range(len(actions)):
            for j in range(len(actions[i])):
                if count >= self.MAX_ACTIONS:
                    break
                action_vector.append(encode_rule(actions[i][j]))
                count += 1
            if count >= self.MAX_ACTIONS:
                break

        rule_dim = len(action_vector[0]) if action_vector else 0
        while len(action_vector) < self.MAX_ACTIONS:
            action_vector.append([0] * rule_dim)

        actions_tensor = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0)
        return chord_tensor, actions_tensor

    def step_nn(self):
        chord_tensor, actions_tensor = self.get_state_tensor()
        model_input = self.concat_state_and_actions(chord_tensor, actions_tensor)
        results = self.model.forward(model_input)
        index_map = self.build_action_index_map()[0:self.MAX_ACTIONS]
        # print(len(index_map))
        if len(index_map) == 0:
            return None  # no actions possible
        
        q_values = self.model(model_input).detach().squeeze(0)[0:len(index_map)]  # shape: (num_actions,)
        flat_index = q_values.squeeze(0).argmax().item()

        # reward_before = self.evaluate_tree()
        i, j = index_map[flat_index]
        reward_before = self.evaluate_tree()
        self.apply_rule_index(i, j)
        reward_after = self.evaluate_tree()
        # reward_after = self.evaluate_tree()
        last_rule = self.applied_rules[-1]
        # reward = prob_dict.get(last_rule.make_hashable(), 1e-10)  # Use same small value as in evaluate_tree
        # delta_reward = reward_after - reward_before
        next_chord_tensor, next_actions_tensor = self.get_state_tensor()
        # reward = self.evaluate_tree() if self.is_terminal() else np.log(prob_dict.get(last_rule.make_hashable(), 1e-10))
        reward = self.evaluate_tree() if self.is_terminal() else np.log(self.prob_dict.get(last_rule.make_hashable(), 1e-10))
        # reward = reward_after - reward_before
        done = self.is_terminal()
        self.actions = self.get_actions()
        # if done:
        #     delta_reward = self.evaluate_tree()
        return {
        "state": chord_tensor,
        "actions": actions_tensor,
        "action_index": flat_index,
        "reward": reward,
        "next_state": next_chord_tensor,
        "next_actions": next_actions_tensor,
        "done": done
    }
    def step(self, epsilon=0.1):
        chord_tensor, actions_tensor = self.get_state_tensor()
        model_input = self.concat_state_and_actions(chord_tensor, actions_tensor)
        results = self.model.forward(model_input)
        index_map = self.build_action_index_map()[0:self.MAX_ACTIONS]
        # print(len(index_map))
        if len(index_map) == 0:
            return None  # no actions possible
        
        if random.random() < epsilon:
            flat_index = random.randint(0, len(index_map) - 1)
        else:
            q_values = self.model(model_input).detach().squeeze(0)[0:len(index_map)]  # shape: (num_actions,)
            flat_index = q_values.squeeze(0).argmax().item()

        # reward_before = self.evaluate_tree()
        i, j = index_map[flat_index]
        reward_before = self.evaluate_tree()
        self.apply_rule_index(i, j)
        reward_after = self.evaluate_tree()
        # reward_after = self.evaluate_tree()
        last_rule = self.applied_rules[-1]
        # reward = prob_dict.get(last_rule.make_hashable(), 1e-10)  # Use same small value as in evaluate_tree
        # delta_reward = reward_after - reward_before
        next_chord_tensor, next_actions_tensor = self.get_state_tensor()
        # reward = self.evaluate_tree() if self.is_terminal() else np.log(prob_dict.get(last_rule.make_hashable(), 1e-10))
        reward = self.evaluate_tree() if self.is_terminal() else np.log(self.prob_dict.get(last_rule.make_hashable(), 1e-10))
        # reward = reward_after - reward_before
        done = self.is_terminal()
        self.actions = self.get_actions()
        # if done:
        #     delta_reward = self.evaluate_tree()
        return {
        "state": chord_tensor,
        "actions": actions_tensor,
        "action_index": flat_index,
        "reward": reward,
        "next_state": next_chord_tensor,
        "next_actions": next_actions_tensor,
        "done": done
    }
    

    def nn_simulate(self, epsilon=0.1):
        while not self.is_terminal():
            result = self.step_nn()
            if result is None:
                break
 

    def add_model(self, model):
        self.model = model

    def is_terminal(self):
        return len(self.current_state) == 1

    def random_baseline_step(self):
        chord_tensor, actions_tensor = self.get_state_tensor()
        index_map = self.build_action_index_map()

        if len(index_map) == 0:
            return None
        flat_index = random.randint(0, len(index_map) - 1)
        i, j = index_map[flat_index]
        self.apply_rule_index(i, j)

    def simulate_random_baseline(self):
        while not self.is_terminal():
            self.random_baseline_step()

    def simulate_greedy_baseline(self):
        while not self.is_terminal():
            actions = self.get_actions()
            index_map = self.build_action_index_map()
            max_prob = float('-inf')
            max_i = 0
            max_j = 0
            for i, rule_list in enumerate(actions):
                for j, rule in enumerate(rule_list):
                    prob = self.prob_dict.get(rule.make_hashable(), 1e-10)  # Use same small value as in evaluate_tree
                    if prob > max_prob:
                        max_prob = prob
                        max_i = i
                        max_j = j
            if len(index_map) == 0:
                break
            self.apply_rule_index(max_i, max_j)

    def evaluate_tree(self):
        log_probs = 0
        for rule in self.applied_rules:
            log_probs += np.log(self.prob_dict.get(rule.make_hashable(), 1e-10))  # Add a small value to avoid log(0)

        return log_probs 

