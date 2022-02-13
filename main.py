from KMeans import KMeans
from cosine_similarity import cosine_similarity_method
from euclidean_distance import euclidean_distance_method
from utils import read_csv_file, calc_average, normalize_vector, normalize_data, calc_euclidean_dist, \
    calc_cosine_similarity, write_to_file

if __name__ == '__main__':
    data = read_csv_file('dataset.csv')
    euclidean_distance_method(data)
    cosine_similarity_method(data)


