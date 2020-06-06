import ml.decision_tree.decision_trees.SplitData as sd
import ml.decision_tree.decision_trees.RecursiveSplitting as rs
import pprint

def build_tree(train, max_depth, min_size):
    root = sd.get_split(train)
    rs.recursive_split(root, 1, max_depth, min_size)
    return root
