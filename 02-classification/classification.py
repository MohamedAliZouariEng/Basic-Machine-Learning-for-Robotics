from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt 
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_reduced = X[:, [0,1]]

feature_1_name = iris.feature_names[0]
feature_2_name = iris.feature_names[1]

X_train_red, X_test_red, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=3850)
model.fit(X_train_red, y_train)

y_pred_red = model.predict(X_test_red)

accuracy_red = accuracy_score(y_test, y_pred_red)
precision_red = precision_score(y_test, y_pred_red, average='macro')
recall_red = recall_score(y_test, y_pred_red, average='macro')
f1_red = f1_score(y_test, y_pred_red, average='macro')
conf_matrix_red = confusion_matrix(y_test, y_pred_red)


print(f'Accuracy: {accuracy_red:.2f}')
print(f'Precision: {precision_red:.2f}')
print(f'Recall: {recall_red:.2f}')
print(f'F1 Score:: {f1_red:.2f}')
print('Confusion Matrix:')
print(conf_matrix_red)

def plot_decision_boundary_with_images(X, y, model, title, xlabel, ylabel, img_paths):
    h = .02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(figsize=(10, 6))

    Z = model.predict((np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.scatter(X[:,0], X[:,1], c=y, edgecolors='k', marker='o', s=10, alpha=0.2)

    img_class_0 = mpimg.imread(img_paths[0])
    img_class_1 = mpimg.imread(img_paths[1])
    img_class_2 = mpimg.imread(img_paths[2])

    zoom_level = 1.0 * fig.get_dpi() / 100

    img_offset_0 = OffsetImage(img_class_0, zoom=zoom_level)
    img_offset_1 = OffsetImage(img_class_1, zoom=zoom_level)
    img_offset_2 = OffsetImage(img_class_2, zoom=zoom_level)

    for i in range(len(X)):
        fontsize = 4 * fig.get_dpi() / 50
        if y[i] == 0:
            img_box = AnnotationBbox(img_offset_0, (X[i,0], X[i,1]), frameon=False)
        elif y[i] == 1:
            img_box = AnnotationBbox(img_offset_1, (X[i,0], X[i,1]), frameon=False)
        else:
            img_box = AnnotationBbox(img_offset_2, (X[i,0], X[i,1]), frameon=False)
        
        ax.add_artist(img_box)

        if y_pred_red[i] == y[i]:
            annotation = 'OK'
            color = "green"
            ax.text(X[i,0] - 0.05, X[i,1] + 0.3, annotation, color=color, fontsize=fontsize + 2, weight='bold')
        else:
            annotation = 'WRONG'
            color = "red"
            ax.text(X[i,0] - 0.1, X[i,1] + 0.3, annotation, color=color, fontsize=fontsize, weight='bold')
    plt.show()


img_paths = {0: "./images/iris_setosa.png",
             1: "./images/iris_versicolor.png",
             2: "./images/iris_virginica.png"}

plot_decision_boundary_with_images(X_test_red, y_test, model, "Logistic Regression Decision Boundary with Images", feature_1_name, feature_2_name, img_paths)




"""
OUTPUT
Accuracy: 0.82
Precision: 0.81
Recall: 0.79
F1 Score:: 0.79
Confusion Matrix:
[[19  0  0]
 [ 0  7  6]
 [ 0  2 11]]
"""


