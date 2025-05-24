import random
import numpy as np
from Chord import Chord
from Helpers import get_possible_intervals, get_qualities, get_possible_rhs_from_str, get_applicable_rules
from Rule import Rule
from Chord import Chord

class TreeNode:
    def __init__(self, chord):
        self.chord = chord
        self.left = None           # child node
        self.right = None          # child node
        self.parent = None         # to be filled in later

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        return f"TreeNode(label={self.chord.label})"
    def to_dict(self):
        """
        Recursively converts the TreeNode and its children into a dictionary representation.
        """
        return {
            "chord": self.chord.label,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None
        }

def create_parent_node(left, right, parent, parent_quality):
    parent = TreeNode(Chord(f"{parent.chord.label}"))
    parent.left = left
    parent.right = right
    left.parent = parent
    right.parent = parent

    return parent

def get_top_level_nodes(nodes):
    """
    Given a list of TreeNodes (leaves and partial trees), 
    return the current top-level (root) nodes that are not children of any other node.
    """
    return [node for node in nodes if node.parent is None]
