import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


def gen_ring(r, var, num):
    r_array = np.random.normal(r, var, num)
    t_array = [np.random.random() * 2 * np.math.pi for i in range(num)]
    data = [[r_array[i] * np.math.cos(t_array[i]),
             r_array[i] * np.math.sin(t_array[i])]
            for i in range(num)]
    return data


def gen_gauss(mean, cov, num):
    return np.random.multivariate_normal(mean, cov, num)


def gen_clusters():
    data = gen_ring(1, 0.1, 100)
    data = np.append(data, gen_ring(3, 0.1, 300), 0)
    data = np.append(data, gen_ring(5, 0.1, 500), 0)
    mean = [7, 7]
    cov = [[0.5, 0], [0, 0.5]]
    data = np.append(data, gen_gauss(mean, cov, 100), 0)
    return np.round(data, 4)

def show_scatter(data,colors):
    cm = plt.cm.get_cmap('Spectral')
    x,y = data.T
    plt.scatter(x,y,c=colors,cmap=cm)
    plt.axis()
    plt.xlabel("x")
    plt.ylabel("y")


data = gen_clusters()
db = DBSCAN(eps=0.5, min_samples=5).fit(data)

labels = set(db.labels_)
n_clusters = len(labels) - (1 if -1 in labels else 0)
noise_mask = (db.labels_ == -1)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(labels))]

plt.title('Estimated number of clusters: %d' % n_clusters)
show_scatter(data[~noise_mask],db.labels_[~noise_mask])
show_scatter(data[noise_mask],'k')
plt.show()
#
# y_pred_0 = DBSCAN(eps=0.5, min_samples=4).fit_predict(data)
# plt.scatter(data[:,0], data[:,1], c=y_pred_0)
# plt.show()

y_pred_1 = KMeans(n_clusters=4).fit_predict(data)
plt.scatter(data[:,0], data[:,1], c=y_pred_1)
plt.show()

y_pred_2 = AgglomerativeClustering(n_clusters=4).fit_predict(data)
plt.scatter(data[:,0], data[:,1], c=y_pred_2)
plt.show()
