import pandas as pd
import math

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
        
    def add_child(self, value, node):
        self.children[value] = node


def entropy(data):
    count = data['PlayTennis'].value_counts()
    total = len(data)
    entropy = 0
    
    for value in count:
        p = value / total
        entropy += -p * math.log2(p)
    
    return entropy


def information_gain(data, attribute):
    total_entropy = entropy(data)
    count = data[attribute].value_counts()
    weighted_entropy = 0
    
    for value, value_count in count.items():
        subset = data[data[attribute] == value]
        subset_entropy = entropy(subset)
        weighted_entropy += (value_count / len(data)) * subset_entropy
    
    return total_entropy - weighted_entropy


def get_best_attribute(data, attributes):
    best_attribute = None
    max_gain = -1
    
    for attribute in attributes:
        gain = information_gain(data, attribute)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
    
    return best_attribute


def build_tree(data, attributes):
    play_tennis = data['PlayTennis']
    
    if len(set(play_tennis)) == 1:
        leaf = Node(play_tennis.iloc[0])
        return leaf
    
    if len(attributes) == 0:
        values, counts = np.unique(play_tennis, return_counts=True)
        index = np.argmax(counts)
        leaf = Node(values[index])
        return leaf
    
    best_attribute = get_best_attribute(data, attributes)
    root = Node(best_attribute)
    
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if len(subset) == 0:
            values, counts = np.unique(play_tennis, return_counts=True)
            index = np.argmax(counts)
            leaf = Node(values[index])
            root.add_child(value, leaf)
        else:
            child = build_tree(subset, remaining_attributes)
            root.add_child(value, child)
    
    return root


def classify(instance, tree):
    while tree.children:
        attribute = tree.attribute
        value = instance[attribute]
        if value not in tree.children:
            return None
        tree = tree.children[value]
    
    return tree.attribute


def main():
    # Load data from CSV
    data = pd.read_csv('data.csv')
    
    # Get attribute names
    attributes = list(data.columns[1:-1])
    
    # Build decision tree
    tree = build_tree(data, attributes)
    
    # Classify a new sample
    #  new_sample = {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak'}   #output--no
    new_sample = {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'}  #output --yes
    classification = classify(new_sample, tree)
    print(new_sample)
    
    print("Classification:", classification)

if __name__ == '__main__':
    main()
