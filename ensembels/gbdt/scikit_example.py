from treeg_gbdt import GradientBoostedTreeGClassifier, GradientBoostedTreeGRegressor
from treeg.graph_treeg.data_formetter_graph_level import DataFormatter
from treeg.graph_treeg.graph_data_graph_level import GraphData
from experiments import datasets
import numpy as np
from sklearn.model_selection import train_test_split

dataset = datasets.TU_MUTAG()

formatter = DataFormatter(GraphData)
X, y = formatter.pyg_data_list_to_tree_graph_data_list(dataset)
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf = GradientBoostedTreeGClassifier(n_estimators=50, learning_rate=0.1, max_depth=10, random_state=0).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print('score: ' + str(score))

feature_importance = clf.feature_importances_
print('feature_importance: ' + str(feature_importance))