from sklearn.cluster import KMeans
from sklearn import datasets
import os
import matplotlib.pyplot as plt
import numpy as np

#Function to cluster the readings from all energy meters based on similarities in the energy consumtpion
def FindOptimalClusters(data, cluster_rng, my_path):

    #Create a variable to store the associated error (Sum of Square error) values for each fitted model
    sse = []

    #Determine how many clusters are needed to represent the data
    #Max number of clusters is determined by user defined value
    for i in range(1,cluster_rng):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
        
        #Append the error
        sse.append(kmeans.inertia_)

    #Plot the SSE as a function of the k-means clusters in order to visualize what number of clusters is best
    fig = plt.figure()

    plt.plot(range(1,cluster_rng), sse)
    plt.xlabel("Clusters")
    plt.ylabel("SSE")

    fig.savefig(my_path + "KMeansClusterPlot.png")


    
def clustData(data,labels, num_clusters):

    #Perform kmeans clustering given the optimal number of clusters
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++")
    kmeans = kmeans.fit(np.gradient(data[:,:],axis=1))


    #Store the cluster centers
    cluster_means = kmeans.cluster_centers_.tolist()

    #Aggregate all the data together based on the kmeans labels (corresponds to the cluster each data point was projected to)
    dataClusters = [[] for i in range(num_clusters)]
    dataLabels = [[] for i in range(num_clusters)]

    #Initialize a variable to keep track of data array index
    count = 0
    for i in kmeans.labels_:
        curMeter = data[count,:].tolist()
        dataClusters[i].append(curMeter)
        dataLabels[i].append(labels[count])
        count = count + 1
    
    return(dataClusters, dataLabels, cluster_means)



