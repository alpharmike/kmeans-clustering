from KMeans import KMeans
from utils import write_to_file, calc_cosine_similarity
import math


def cosine_similarity_method(data):
    '''a KMeans model is created, clusters are obtained by the fit model, and the parameter s is calculated and written
    into a file'''
    model = KMeans(k=2, max_iter=1000, tolerance=0.001, loss='cosine_similarity')
    clusters = model.fit(data)
    value = find_s(clusters)
    # print(clusters)
    # print(value)
    write_to_file(data, clusters, value, 'cosine_similarity', 'cosine_sim.txt')


def find_s(clusters):
    '''we calculate cosine the similarity of each point in one cluster with all points of the other cluster, and the
    maximum value will be the parameter s'''
    cluster0 = clusters[0]
    cluster1 = clusters[1]
    s = 0

    for i in range(len(cluster0)):
        for j in range(len(cluster1)):
            s = max(s, calc_cosine_similarity(cluster0[i], cluster1[j]))
    return s