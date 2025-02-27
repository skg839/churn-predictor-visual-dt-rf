import numpy as np
import pandas as pd
import os
import graphviz
import cairosvg
import tkinter as tk
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tkinter import messagebox
from imblearn.over_sampling import RandomOverSampler

class TreeNode:
    def __init__(self, split_feature=None, split_value=None, left_child=None, right_child=None, *, label=None):
        self.split_feature = split_feature   # Index of the feature used for splitting
        self.split_value = split_value       # Threshold for the split
        self.left_child = left_child         # Left branch (data <= threshold)
        self.right_child = right_child       # Right branch (data > threshold)
        self.label = label                   # Class label if it's a terminal node

    def is_leaf(self):
        return self.label is not None

class CustomDecisionTree:
    def __init__(self, min_samples=2, max_depth=50, n_features=None):
        self.min_samples = min_samples       # Minimum data points to allow a split
        self.max_depth = max_depth           # Maximum depth of the tree
        self.n_features = n_features         # How many features to consider for each split
        self.root = None                     # Root of the tree

    def fit(self, X, y):
        # Set the number of features to check if not provided
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        num_samples, num_feats = X.shape
        unique_labels = np.unique(y)

        # Stop if the maximum depth is reached, or node is pure, or not enough samples exist
        if current_depth >= self.max_depth or len(unique_labels) == 1 or num_samples < self.min_samples:
            return TreeNode(label=self._majority_class(y))

        # Randomly choose features for this node
        feature_indices = np.random.choice(num_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._choose_best_split(X, y, feature_indices)

        # If no good split is found, create a leaf
        if best_feature is None:
            return TreeNode(label=self._majority_class(y))
        
        left_idx, right_idx = self._split_dataset(X[:, best_feature], best_threshold)
        # If the split produces an empty branch, also create a leaf
        if len(left_idx) == 0 or len(right_idx) == 0:
            return TreeNode(label=self._majority_class(y))
        
        # Recursively build left and right subtrees
        left_branch = self._build_tree(X[left_idx, :], y[left_idx], current_depth + 1)
        right_branch = self._build_tree(X[right_idx, :], y[right_idx], current_depth + 1)
        return TreeNode(split_feature=best_feature, split_value=best_threshold,
                        left_child=left_branch, right_child=right_branch)

    def _choose_best_split(self, X, y, feature_indices):
        best_info_gain = -1
        best_feat = None
        best_thresh = None

        for feat in feature_indices:
            column = X[:, feat]
            # Consider every unique threshold in the column
            for thresh in np.unique(column):
                info_gain = self._calculate_info_gain(y, column, thresh)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh

    def _calculate_info_gain(self, y, column, threshold):
        # Compute entropy of current set
        base_entropy = self._compute_entropy(y)
        left_indices, right_indices = self._split_dataset(column, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        # Compute weighted entropy of the split branches
        total = len(y)
        left_entropy = self._compute_entropy(y[left_indices])
        right_entropy = self._compute_entropy(y[right_indices])
        weighted_entropy = (len(left_indices) / total) * left_entropy + (len(right_indices) / total) * right_entropy
        return base_entropy - weighted_entropy

    def _split_dataset(self, column, threshold):
        left_idx = np.argwhere(column <= threshold).flatten()
        right_idx = np.argwhere(column > threshold).flatten()
        return left_idx, right_idx

    def _compute_entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        # Only consider non-zero probabilities to avoid log(0)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        # For each sample, traverse the tree to get a prediction.
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, sample, node):
        # Recursively traverse the tree until a leaf node is reached.
        if node.is_leaf():
            return node.label
        if sample[node.split_feature] <= node.split_value:
            return self._traverse(sample, node.left_child)
        return self._traverse(sample, node.right_child)

    def export_tree_real_png(self, features, class_labels, file_base="tree_visual"):
        
        #Generates a visualization of the decision tree
        #Saves an SVG file and then converts that SVG into a PNG
        
        if self.root is None:
            raise ValueError("The tree hasn't been trained yet.")

        diagram = graphviz.Digraph()

        # Use a high DPI for quality output
        diagram.graph_attr.update({'dpi': '600'})

        def add_node_edges(node, parent_id=None, edge_lbl=""):
            if node is None:
                return
            if node.is_leaf():
                node_label = f"Class: {class_labels[node.label]}"
                diagram.node(str(id(node)), node_label, shape="box", style="filled", fillcolor="lightgray")
            else:
                node_label = f"{features[node.split_feature]} ≤ {node.split_value:.2f}"
                diagram.node(str(id(node)), node_label, shape="ellipse", style="filled", fillcolor="lightblue")
            if parent_id is not None:
                diagram.edge(parent_id, str(id(node)), label=edge_lbl)
            add_node_edges(node.left_child, str(id(node)), "≤")
            add_node_edges(node.right_child, str(id(node)), ">")

        add_node_edges(self.root)

        # Get the SVG data for the tree diagram.
        svg_output = diagram.pipe(format='svg')
        svg_filename = f"{file_base}.svg"

        with open(svg_filename, "wb") as f:
            f.write(svg_output)

        print(f"SVG file created: {svg_filename}")

        png_filename = f"{file_base}.png"
        cairosvg.svg2png(bytestring=svg_output, write_to=png_filename,
                         output_width=8000, output_height=6000)
        print(f"PNG file created: {png_filename}")
        messagebox.showinfo("Export Complete", f"Visualizations saved:\n{svg_filename}\n{png_filename}")
        return svg_output

class CustomRandomForest:
    def __init__(self, num_trees=5, max_depth=50, min_split=2, n_features=None):
        self.num_trees = num_trees          # Total trees in the ensemble
        self.max_depth = max_depth          # Maximum allowed depth per tree
        self.min_split = min_split          # Minimum number of samples required to split
        self.n_features = n_features        # Number of features to consider per tree
        self.trees = []                     # Container for the trees

    def fit(self, X, y):
        self.trees = []
        total_samples = X.shape[0]
        # Build each tree with a bootstrap sample of the data
        for _ in range(self.num_trees):
            tree = CustomDecisionTree(max_depth=self.max_depth,
                                      min_samples=self.min_split,
                                      n_features=self.n_features)
            indices = np.random.choice(total_samples, total_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from each tree and vote on the final output
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Transpose so each row corresponds to a sample's predictions
        tree_votes = np.swapaxes(tree_predictions, 0, 1)
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_votes])

try:
    data = pd.read_csv("customer_booking.csv")
except UnicodeDecodeError:
    data = pd.read_csv("customer_booking.csv", encoding="latin1")
data.fillna(method='ffill', inplace=True)

if 'churn' not in data.columns:
    raise ValueError("The 'churn' target column is missing.")

# Separate features from target.
features_df = data.drop("churn", axis=1)
target = data["churn"]

# One-hot encode categorical variables.
features_df = pd.get_dummies(features_df, drop_first=True)
features_list = features_df.columns.tolist()

if target.dtype == 'O':
    target, _ = pd.factorize(target)

X_all = features_df.values
y_all = target.values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=64)

# Use RandomOverSampler to balance the training set.
oversampler = RandomOverSampler(random_state=64)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

def compute_accuracy(true_vals, pred_vals):
    return np.sum(true_vals == pred_vals) / len(true_vals)

def train_decision_tree():
    print("Training Decision Tree...")
    model = CustomDecisionTree()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = compute_accuracy(y_test, preds)
    print("Decision Tree Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    return model

def train_random_forest():
    print("Training Random Forest...")
    model = CustomRandomForest()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = compute_accuracy(y_test, preds)
    print("Random Forest Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    return model

def launch_gui():
    window = tk.Tk()
    window.title("Model Selection")

    selected_model = tk.StringVar(value="decision_tree")

    tk.Label(window, text="Select Model to Run:").pack(pady=5)
    tk.Radiobutton(window, text="Decision Tree", variable=selected_model, value="decision_tree").pack(anchor="w", padx=10)
    tk.Radiobutton(window, text="Random Forest", variable=selected_model, value="random_forest").pack(anchor="w", padx=10)

    def execute_model():
        model_type = selected_model.get()
        if model_type == "decision_tree":
            dt = train_decision_tree()
            # Export tree visualization (both SVG and PNG)
            dt.export_tree_real_png(features_list, ["No Churn", "Churn"])
        elif model_type == "random_forest":
            train_random_forest()
            messagebox.showinfo("Run Complete", "Random Forest executed. No visualization available.")
        else:
            messagebox.showerror("Error", "Unrecognized model selection.")

    tk.Button(window, text="Run Model", command=execute_model).pack(pady=10)
    window.mainloop()

if __name__ == "__main__":
    launch_gui()