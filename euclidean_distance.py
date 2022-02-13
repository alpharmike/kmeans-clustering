from KMeans import KMeans
from utils import write_to_file, calc_euclidean_dist
import math


def euclidean_distance_method(data):
    '''a KMeans model is created, clusters are obtained by the fit model, and the parameter s is calculated and written
        into a file'''
    model = KMeans(k=2, max_iter=1000, tolerance=0.001, loss='euclidean_dist')
    clusters = model.fit(data)
    value = find_d(clusters)
    # print(clusters[0])
    # print(clusters[1])
    # print(value)
    write_to_file(data, clusters, value, 'euclidean_dist', 'euclid_dist')


def find_d(clusters):
    cluster0 = clusters[0]
    cluster1 = clusters[1]
    d = math.inf
    for i in range(len(cluster0)):
        for j in range(len(cluster1)):
            d = min(d, calc_euclidean_dist(cluster0[i], cluster1[j]))
    return d


def find_max_dist(cluster):
    '''we calculate the distance of each point in one cluster with all points of the other cluster, and the maximum
    value will be the parameter d'''
    a = 0
    for i in range(len(cluster) - 1):
        for j in range(i + 1, len(cluster)):
            a = max(a, calc_euclidean_dist(cluster[i], cluster[j]))
    return a
