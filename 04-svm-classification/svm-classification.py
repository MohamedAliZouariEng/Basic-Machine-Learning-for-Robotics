import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg


iris = load_iris()
X = iris.data[:, :2]
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

rbf_svm = SVC(kernel='rbf', gamma='scale')
rbf_svm.fit(X_train, y_train)

linear_predictions = linear_svm.predict(X_test)
rbf_predictions = rbf_svm.predict(X_test)

print('Linear SVM Classification Report:')
print(classification_report(y_test, linear_predictions))

print('RBF SVM Classification Report:')
print(classification_report(y_test, rbf_predictions))

def plot_decision_boundary_with_images(model,X, y, y_pred, title, img_paths):
    h = .02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(figsize=(10, 6))

    Z = model.predict((np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)

    img_setosa = mpimg.imread(img_paths[0])
    img_versicolor = mpimg.imread(img_paths[1])
    img_virginica = mpimg.imread(img_paths[2])

    zoom_level = 1.0 * fig.get_dpi() / 100

    img_offset_0 = OffsetImage(img_setosa, zoom=zoom_level)
    img_offset_1 = OffsetImage(img_versicolor, zoom=zoom_level)
    img_offset_2 = OffsetImage(img_virginica, zoom=zoom_level)

    for i in range(len(X)):
        fontsize = 4 * fig.get_dpi() / 50
        if y[i] == 0:
            img_box = AnnotationBbox(img_offset_0, (X[i,0], X[i,1]), frameon=False)
        elif y[i] == 1:
            img_box = AnnotationBbox(img_offset_1, (X[i,0], X[i,1]), frameon=False)
        else:
            img_box = AnnotationBbox(img_offset_2, (X[i,0], X[i,1]), frameon=False)
        
        ax.add_artist(img_box)

        if y_pred[i] == y[i]:
            annotation = 'OK'
            color = "green"
            ax.text(X[i,0] - 0.05, X[i,1] + 0.3, annotation, color=color, fontsize=fontsize + 2, weight='bold')
        else:
            annotation = 'WRONG'
            color = "red"
            ax.text(X[i,0] - 0.15, X[i,1] + 0.3, annotation, color=color, fontsize=fontsize, weight='bold')
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")
    ax.set_title(title)
    plt.show()


img_paths = {0: "./images/iris_setosa.png",
             1: "./images/iris_versicolor.png",
             2: "./images/iris_virginica.png"}

plot_decision_boundary_with_images(linear_svm, X_train, y_train, linear_svm.predict(X_train), "Linear SVM Decision Boundary", img_paths)
plot_decision_boundary_with_images(rbf_svm, X_train, y_train, rbf_svm.predict(X_train), "RBF SVM Decision Boundary", img_paths)

"""
OUTPUT 
Linear SVM Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       0.70      0.54      0.61        13
           2       0.62      0.77      0.69        13

    accuracy                           0.80        45
   macro avg       0.78      0.77      0.77        45
weighted avg       0.81      0.80      0.80        45

RBF SVM Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       0.54      0.54      0.54        13
           2       0.54      0.54      0.54        13

    accuracy                           0.73        45
   macro avg       0.69      0.69      0.69        45
weighted avg       0.73      0.73      0.73        45
"""