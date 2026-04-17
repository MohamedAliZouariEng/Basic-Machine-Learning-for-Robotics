import numpy as np 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
X_3d = np.random.rand(100, 3)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c='b', marker='o')
ax.set_title('Original 3D Dataset')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_3d)

plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c='r', marker='o')
plt.title('2D Projection of 3D Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()