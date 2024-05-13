import numpy as np
import math

def initialize_centroids_forgy(data, k):
    randomCentroidsIndex = np.arange(data.shape[0])
    randomCentroidsIndex = np.random.choice(randomCentroidsIndex, k, replace=False)
    randomCentroids = data[randomCentroidsIndex]
    return randomCentroids

def initialize_centroids_kmeans_pp(data, k):
    def distanceBeatweenNodes(x, y):
        distance = 0
        for i in range(len(x)):
            distance += (x[i] - y[i])**2  
            
        return math.sqrt(distance)
    
    centroids = []
    centroids.append(data[np.random.randint(0, data.shape[0])])
    for i in range(k - 1):
        furthestDistance = 0
        for node in data:
            distance = 0
            for centroid in centroids:
                distance += distanceBeatweenNodes(centroid, node)
            if furthestDistance < distance:
                furthestDistance = distance
                furthestNode = node
        centroids.append(furthestNode)
        
    return centroids

def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    return None

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    return None

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

