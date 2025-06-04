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
    
    def get_total_depth(self):
        """
        Returns the total depth (height) of the subtree rooted at this node.
        Defined as the number of edges on the longest path from this node to a leaf.
        """
        if self.is_leaf():
            return 0
        left_depth = self.left.get_total_depth() if self.left else 0
        right_depth = self.right.get_total_depth() if self.right else 0
        return 1 + max(left_depth, right_depth)
    
    def get_edit_distance(self, other):
        """
        Calculate the edit distance between this tree and another tree.
        Edit distance is defined as the minimum number of operations required to transform one tree into another.
        """

    def tree_edit_distance(self, other):
        """
        Computes a simplified tree edit distance between this tree and another tree.
        Edit operations considered: insert, delete, and label change.
        """

        # Handle missing nodes at the top level
        if self is None and other is None:
            return 0
        if self is None:
            return other.size()
        if other is None:
            return self.size()

        # Cost of label change
        label_cost = 0 if self.chord.label == other.chord.label else 1

        # Safely compute left subtree distance
        if self.left and other.left:
            left_cost = self.left.tree_edit_distance(other.left)
        elif self.left:
            left_cost = self.left.size()
        elif other.left:
            left_cost = other.left.size()
        else:
            left_cost = 0

        # Safely compute right subtree distance
        if self.right and other.right:
            right_cost = self.right.tree_edit_distance(other.right)
        elif self.right:
            right_cost = self.right.size()
        elif other.right:
            right_cost = other.right.size()
        else:
            right_cost = 0

        return label_cost + left_cost + right_cost
    def size(self):
        """
        Returns the total number of nodes in the subtree rooted at this node.
        Used in edit distance to estimate insert/delete costs.
        """
        left_size = self.left.size() if self.left else 0
        right_size = self.right.size() if self.right else 0
        return 1 + left_size + right_size

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
