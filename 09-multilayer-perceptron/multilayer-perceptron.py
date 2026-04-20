import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg 

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_mlp)
precision = precision_score(y_test, y_pred_mlp)
recall = recall_score(y_test, y_pred_mlp)
f1 = f1_score(y_test, y_pred_mlp)
conf_matrix = confusion_matrix(y_test, y_pred_mlp)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score:: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

def plot_decision_boundary_with_images(X, y, y_pred, model, title, img_paths):
    h = .02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(figsize=(10, 6))

    Z = model.predict((np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)

    

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

        if y_pred_mlp[i] == y[i]:
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

plot_decision_boundary_with_images(X_test, y_test, y_pred_mlp, mlp, "MLP Decision Boundary with Images",  img_paths)

"""
OUTPUT
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1 Score:: 1.00
Confusion Matrix:
[[17  0]
 [ 0 13]]

"""