import csv
import math
import pandas as pd
from statistics import mean


def read_csv_file(file_path):
    # reading csv file and storing sample in a list
    dataset = pd.read_csv(file_path)
    samples = [[float(key) for key in list(dataset.keys())]]
    for sample in pd.DataFrame(dataset).values:
        samples.append(list(sample))
    return samples


def calc_euclidean_dist(sample1, sample2):
    # calculating euclidean distance of two samples
    euclidean_dist = 0.0
    for i in range(len(sample1)):
        euclidean_dist += pow(sample1[i] - sample2[i], 2)
    return math.sqrt(euclidean_dist)


def calc_dot_product(sample1, sample2):
    dot_product = 0.0
    for i in range(len(sample1)):
        dot_product += sample1[i] * sample2[i]
    return dot_product


def calc_norm(sample):
    return math.sqrt(sum([element ** 2 for element in sample]))


def calc_cosine_similarity(sample1, sample2):
    # cosine similarity calculated by the formula (a.b / (||a|| * ||b||) where a, b are vectors)
    return calc_dot_product(sample1, sample2) / (calc_norm(sample1) * calc_norm(sample2))


def calc_average(samples):
    # calculating the average of each feature in a set of samples
    avg_sample = []
    if len(samples) != 0:
        for i in range(len(samples[0])):
            elements = []
            for sample in samples:
                elements.append(sample[i])
            avg_sample.append(mean(elements))
    return avg_sample


def normalize_vector(vector):
    norm = calc_norm(vector)
    for index in range(len(vector)):
        vector[index] /= norm
    return vector


def normalize_data(data):
    for index in range(len(data)):
        data[index] = normalize_vector(data[index])
    return data


def sum_of_squared_errors(sample1, sample2):
    sqe = 0
    for index in range(len(sample1)):
        sqe += (sample1[index] - sample2[index]) ** 2
    return sqe


def write_to_file(dataset, clusters, value, criterion, filename):
    # criterion is a string indicating either cosine_similarity or euclidean_distance
    # we also use criterion as the filename we create
    # value is either d(if euclidean distance) or s(if cosine similarity)
    a = "sds"
    if not filename.endswith(".txt"):
        filename = filename + ".txt"
    with open(filename, mode='w') as file:
        for sample in dataset:
            if sample in clusters[0]:
                file.write("0\n")
            else:
                file.write("1\n")

        file.write(str(value) + "\n")
    file.close()
