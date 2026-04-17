from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 

import matplotlib.pyplot as plt 

iris = load_iris()
X = iris.data
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf2 = DecisionTreeClassifier(max_depth=2)
clf4 = DecisionTreeClassifier(max_depth=4)
clf5 = DecisionTreeClassifier(max_depth=5)

clf2.fit(X_train, y_train)
clf4.fit(X_train, y_train)
clf5.fit(X_train, y_train)

plt.figure(figsize=(12,8))
tree.plot_tree(clf2, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.savefig("decision_tree_max_depth_2.png")
plt.show()

plt.figure(figsize=(12,8))
tree.plot_tree(clf4, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.savefig("decision_tree_max_depth_4.png")
plt.show()

plt.figure(figsize=(12,8))
tree.plot_tree(clf5, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.savefig("decision_tree_max_depth_5.png")
plt.show()