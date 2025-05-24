import pitchtypes as pt
import ast

class Rule:

    def __init__(self, parent_quality, child_intervals, child_qualities):
        self.parent_quality = parent_quality
        self.child_intervals = child_intervals
        self.child_qualities = child_qualities

    def lhs(self):
        return self.parent_quality
    def rhs(self):
        return self.child_intervals, self.child_qualities

    def __repr__(self):
        return f"Rule(parent_quality={self.parent_quality}, child_intervals={self.child_intervals}, child_qualities={self.child_qualities})"

    def as_dict(self):
        return {
            "parent_quality": self.parent_quality,
            "child_intervals": self.child_intervals,
            "child_qualities": self.child_qualities
        }
    
    @staticmethod
    def from_dict(rule_dict):
        parent_quality = rule_dict.get("parent_quality")
        child_intervals = rule_dict.get("child_intervals")
        child_qualities = rule_dict.get("child_qualities")

        return Rule(parent_quality, child_intervals, child_qualities)

    def make_hashable(self):
        def convert(value):
            if isinstance(value, dict):
                return tuple(sorted((k, convert(v)) for k, v in value.items()))
            elif isinstance(value, list):
                return tuple(convert(v) for v in value)
            else:
                return value

        return tuple(sorted((k, convert(v)) for k, v in self.as_dict().items()))
    
    @staticmethod
    def unhash(hashable_tuple):
        def convert(value):
            if isinstance(value, tuple):
                # Check if it's a dict-like structure (tuple of key-value pairs)
                if all(isinstance(item, tuple) and len(item) == 2 for item in value):
                    return {k: convert(v) for k, v in value}
                else:
                    return [convert(v) for v in value]
            else:
                return value

        return Rule.from_dict({k: convert(v) for k, v in hashable_tuple})
