from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.patches as mpatches

digits = load_digits()
X = digits.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X)

fig, ax = plt.subplots(figsize=(10,8))
scatter = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='tab10', legend=False, ax=ax)
plt.title('K-Means Clustering of Digits Dataset (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')


centroids_original_space = kmeans.cluster_centers_
centroids_pca_space = pca.transform(centroids_original_space)

centroids_images = centroids_original_space.reshape(10,8,8)

for i, (x,y) in enumerate(centroids_pca_space):
    imagebox = OffsetImage(centroids_images[i], cmap='gray', zoom=3.0)
    ab = AnnotationBbox(imagebox, (x,y), frameon=False)
    ax.add_artist(ab)

colors = sns.color_palette('tab10', 10)

x_legend = 1.1
y_legend_start = 0.9
y_offset = 0.08

for i in range(10):
    img = OffsetImage(centroids_images[i], cmap='gray', zoom=3.5)
    ab = AnnotationBbox(img, (x_legend - 0.05, y_legend_start - i * y_offset), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)

    dot_x = x_legend - 0.1
    dot_y = y_legend_start - i * y_offset
    circle = mpatches.Circle((dot_x,dot_y), 0.02, color=colors[i], transform=ax.transAxes, zorder=10)
    ax.add_patch(circle)
plt.show()
