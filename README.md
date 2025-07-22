# Reinforcement Learning for Jazz Harmony Parsing  
This repository was developed as part of the Information Studies Master’s Thesis by **Martin Aviles**, completed on **27/06/2025**.

The project investigates whether a reinforcement learning agent can learn to parse jazz chord sequences using a probabilistic grammar, and outperform a greedy baseline. It includes an environment for syntactic tree construction, a probabilistic grammar for external evaluation, and a deep Q-learning policy for rule selection.

## 📁 Repository Overview

### `main.ipynb`  
- Contains the full training pipeline and the inference logic for model retrieval.  
- Used to generate all the results reported in the thesis.

### `visualisation.ipynb`  
- Supplementary visualizations for analysis and presentation of results.  
- Includes sequential reward plots and tree score comparisons.

## Core Modules

### `environment.py`  
- Defines the RL environment, including:
  - How rules are applied
  - How trees are built
  - Integration with the DQN policy

### `rule.py`  
- Contains logic and structure for the grammar rules used in parsing.  
- Includes constraints and validation checks.

### `TreeNode.py`  
- Implements tree nodes and their manipulation during parsing.  
- Provides utilities for node merging, tree traversal, and structure validation.

### `probabilistic_model.py`  
- Implements a probabilistic context-free grammar model.  
- Used to compute rule probabilities and evaluate full parse trees externally.  
- Estimates rule likelihoods from the annotated treebank.

## 🤖 Deep Q-Learning Components

### `DQN.py`  
- Contains the architecture of the Deep Q-Network used for policy learning.  
- Handles input encoding and action scoring.

### `buffer.py`  
- Experience replay buffer logic.  
- Manages state transitions, sampling, and training batches.

## Utilities

### `helper.py`  
- A set of utility functions used throughout training and evaluation.  
- Includes functions for encoding, logging, and tree formatting.

## Dataset & Rule Definitions

- The full rule set, including interval and quality combinations (with probabilities), is provided as a spreadsheet in the repository.
- Dataset used: **Jazz Harmony Treebank** (150 expert-annotated trees).
- Data split: 80% training / 20% test.



