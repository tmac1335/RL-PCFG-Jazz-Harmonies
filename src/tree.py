import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from pychord import Chord
from copy import deepcopy
from pychord.utils import note_to_val
from scipy import stats
import re

class Tree():
    def __init__(self):


class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children is not None else []

    def __repr__(self):
        return f"Node({self.label!r}, {self.children!r})"


