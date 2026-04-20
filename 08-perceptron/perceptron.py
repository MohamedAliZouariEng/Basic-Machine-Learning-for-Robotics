import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg 

iris = load_iris()
X = iris.data 
y = iris.target

X_reduced = X[:, [0,1]]

feature_1_name = iris.feature_names[0]
feature_2_name = iris.feature_names[1]

X_reduced, y = X_reduced[y != 2], y[y != 2]

X_train_red, X_test_red, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_red = scaler.fit_transform(X_train_red)
X_test_red = scaler.fit_transform(X_test_red)

model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train_red, y_train)

y_pred_red = model.predict(X_test_red)


accuracy_red = accuracy_score(y_test, y_pred_red)
precision_red = precision_score(y_test, y_pred_red)
recall_red = recall_score(y_test, y_pred_red)
f1_red = f1_score(y_test, y_pred_red)
conf_matrix_red = confusion_matrix(y_test, y_pred_red)

print(f'Accuracy: {accuracy_red:.2f}')
print(f'Precision: {precision_red:.2f}')
print(f'Recall: {recall_red:.2f}')
print(f'F1 Score:: {f1_red:.2f}')
print('Confusion Matrix:')
print(conf_matrix_red)



def plot_decision_boundary_with_images(X, y, y_pred, model, title, xlabel, ylabel, img_paths):
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

    ax.scatter(X[:,0], X[:,1], c=y, edgecolors='k', marker='o', s=10, alpha=0.2, cmap=plt.cm.coolwarm)

    img_class_0 = mpimg.imread(img_paths[0])
    img_class_1 = mpimg.imread(img_paths[1])

    zoom_level = 1.0 * fig.get_dpi() / 100

    img_offset_0 = OffsetImage(img_class_0, zoom=zoom_level)
    img_offset_1 = OffsetImage(img_class_1, zoom=zoom_level)

    for i in range(len(X)):
        fontsize = 4 * fig.get_dpi() / 50
        if y[i] == 0:
            img_box = AnnotationBbox(img_offset_0, (X[i,0], X[i,1]), frameon=False)
        else:
            img_box = AnnotationBbox(img_offset_1, (X[i,0], X[i,1]), frameon=False)
        
        ax.add_artist(img_box)

        if y_pred_red[i] == y[i]:
            annotation = 'OK'
            color = "green"
            ax.text(X[i,0] - 0.05, X[i,1] + 0.3, annotation, color=color, fontsize=fontsize + 2, weight='bold')
        else:
            annotation = 'WRONG'
            color = "red"
            ax.text(X[i,0] - 0.1, X[i,1] + 0.3, annotation, color=color, fontsize=fontsize, weight='bold')
    plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'decision_boundary.png'")


img_paths = {0: "images/iris_setosa.png",
             1: "images/iris_versicolor.png"}

plot_decision_boundary_with_images(X_test_red, y_test, y_pred_red, model, "Perceptron Decision Boundary with Images", feature_1_name, feature_2_name, img_paths)

"""
OUTPUT
Accuracy: 0.97
Precision: 0.93
Recall: 1.00
F1 Score:: 0.96
Confusion Matrix:
[[16  1]
 [ 0 13]]

"""