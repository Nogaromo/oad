from tree import DecisiontreeRegressor
import numpy as np


class RandomForest(DecisiontreeRegressor):

    def __init__(self, num_trees=100, min_samples_split=2, max_depth=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []

    def fit(self, X, y):

        for i in range(self.num_trees):
            samples = np.random.choice(a=X.shape[0], size=X.shape[0], replace=True)
            X_s, y_s = X[samples], y[samples]
            clf = DecisiontreeRegressor(X_s, y_s, max_depth=self.max_depth)
            self.decision_trees.append(clf)

    def predict_f(self, X):
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        y = np.array(y).mean(axis=0)
        return y
