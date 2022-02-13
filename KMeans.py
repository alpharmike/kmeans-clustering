from utils import calc_euclidean_dist, calc_cosine_similarity, calc_average, sum_of_squared_errors
import random

loss_functions = {'euclidean_dist': calc_cosine_similarity, 'cosine_similarity': calc_cosine_similarity}


class KMeans:
    # k is the number of clusters we want to partition our samples into
    # max_iter is the maximum number of iteration the KMeans algorithm should be applied to the data
    # tolerance is actually not used here, but can be used as a threshold for sum of squared errors of 2 consecutively calculated centroids
    # loss is the criterion used for the fit method, as a function used for the convergence of the algorithm
    def __init__(self, k, max_iter, tolerance, loss='euclidean_dist'):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.loss = loss

    def fit(self, data, centroids=None):
        # if the provided loss function is not available in our loss_functions, an error is thrown
        if self.loss not in loss_functions:
            raise Exception("Undefined Provided Loss Function")
        if centroids is None:
            centroids = {}
            '''we can initialize centroids randomly, as the commented code indicated, but t might lead us to an
            inaccurate result'''
            # initializing centroids randomly, if not already set
            # for index in range(self.k):
            #     centroids[index] = []
            #     for i in range(len(data[0])):
            #         centroids[index].append(random.randint(0, 5))
            '''here we initialize our centroids with first and second sample of our dataset'''
            centroids[0] = data[0]
            centroids[1] = data[1]
        clusters = {}
        for iter_count in range(self.max_iter):
            '''we save the previously calculated clusters for further comparison with the new one, terminating iteration
            if both are the same'''
            prev_clusters = dict(clusters)
            for index in range(self.k):
                clusters[index] = []
            for sample in data:
                sample_distances_to_centroids = []
                # here we calculate the distance of each sample to the centroids, based on the loss function provided
                for centroid in centroids:
                    sample_distances_to_centroids.append(loss_functions[self.loss](sample, centroids[centroid]))
                    if self.loss == 'euclidean_dist':
                        '''if we use euclidean distance as our criterion, we assign the sample to the cluster with
                        minimum distance'''
                        cluster_index = sample_distances_to_centroids.index(min(sample_distances_to_centroids))
                    else:
                        '''if we use cosine similarity as our criterion, we assign the sample to the cluster with maximum
                        similarity'''
                        cluster_index = sample_distances_to_centroids.index(max(sample_distances_to_centroids))
                clusters[cluster_index].append(sample)

            if prev_clusters == clusters:
                # if clusters don't change in two consecutive iterations, operation is terminated
                print("Operation stopped early at iteration: " + str(iter_count))
                break

            curr_centroids = dict(centroids)

            for cluster in clusters:
                # if operation is not terminated, we update our centroids with the average of newly obtained clusters
                if len(clusters[cluster]) != 0:
                    centroids[cluster] = calc_average(clusters[cluster])

            ''' the commented code below is also a way of terminating the process, using the sum of squared errors of
            the centroids in two consecutive iterations
            '''

            '''updated_centroids = centroids

            terminate = True
            for key in centroids:
                if sum_of_squared_errors(curr_centroids[key], updated_centroids[key]) > self.tolerance:
                    terminate = False
                    break
            if terminate:
                print(iter_count)
                break'''
        if data[0] in clusters[1]:
            new_clusters = dict()
            new_clusters[0] = clusters[1]
            new_clusters[1] = clusters[0]
            return new_clusters
        return clusters
