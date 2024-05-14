import numpy as np
import math

def distanceBeatweenPoints(x, y):
        distance = 0
        for i in range(len(x)):
            distance += (x[i] - y[i])**2  
            
        return math.sqrt(distance)

def initialize_centroids_forgy(data, k):
    randomCentroidsIndex = np.arange(data.shape[0])
    randomCentroidsIndex = np.random.choice(randomCentroidsIndex, k, replace=False)
    randomCentroids = data[randomCentroidsIndex]
    return np.array(randomCentroids)

def initialize_centroids_kmeans_pp(data, k):
    centroids = []
    centroids.append(data[np.random.randint(0, data.shape[0])])
    
    for _ in range(k - 1):
        furthestDistance = 0
        for point in data:
            distance = 0
            for centroid in centroids:
                distance += distanceBeatweenPoints(centroid, point)
            if furthestDistance < distance:
                furthestDistance = distance
                furthestpoint = point
        centroids.append(furthestpoint)
        
    return np.array(centroids)

def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    assignments = []
    for point in data:
        nearestDistance = np.inf
        nearestDistanceCentroid = 0
        for i, centroid in enumerate(centroids):
            distance = distanceBeatweenPoints(point, centroid)
            if nearestDistance > distance:
                nearestDistance = distance
                nearestDistanceCentroid = i
        
        assignments.append(int(nearestDistanceCentroid))
    
    return np.array(assignments)

def update_centroids(data, assignments, centroids):
    # TODO find new centroids based on the assignments
    assigmentsPoints = [[] for _ in range(max(assignments) + 1)]
    for i in range(len(assignments)):
        assigmentsPoints[assignments[i]].append(data[i])
        
    newCentroids = []
    for points in assigmentsPoints:
        if len(points) > 0:
            newCentroids.append(np.mean(points, 0))
        else:
            newCentroids.append(centroids[len(newCentroids)])
    return np.array(newCentroids)

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
        centroids = update_centroids(data, assignments, centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

