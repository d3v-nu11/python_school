import math
import collections
import random

class observation:

    def __init__(self, data, name):
        self.data = data
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)


def distance(p1,p2):
    return sum([(p1[i] - p2[i])**2 for i in range(len(p1))])**0.5


def kmeans(observations,k):
    centroids = [[0 for _ in range(len(observations[0].data))] for _ in range(k)]
    clusters = [[] for _ in range(k)]
    max_iterations = 1000
    iteration = 1
    no_changes = 0

    for ob in observations:
        index = random.randint(0,len(clusters)-1)
        clusters[index].append(ob)
    for i in range(k):
        if clusters[i]:
            for j in range(len(centroids[i])):
                centroids[i][j] = sum(x.data[j] for x in clusters[i]) / len(clusters[i])
    distances = 0
    for i in range(len(clusters)):
        distances += sum(distance(ob.data,centroids[i]) for ob in clusters[i])
    print(f"Iteration {iteration}: {distances}")
    while iteration < max_iterations:
        new_clusters = [[] for _ in range(k)]
        for ob in observations:
            closest_centroid = min(centroids, key=lambda c: distance(ob.data, c))
            cluster_index = centroids.index(closest_centroid)
            new_clusters[cluster_index].append(ob)

        if clusters == new_clusters:
            if no_changes == 2:
                break
            else:
                no_changes += 1
        else:
            no_changes = 0
        clusters = new_clusters

        for i in range(k):
            if clusters[i]:
                for j in range(len(centroids[i])):
                    centroids[i][j] = sum(x.data[j] for x in clusters[i]) / len(clusters[i])

        iteration += 1
        distances = 0
        for i in range(len(clusters)):
            distances += sum(distance(ob.data, centroids[i]) for ob in clusters[i])
        print(f"Iteration {iteration}: {distances}")

    for i in range(len(clusters)):
        if clusters[i]:
            counter_setosa = 0
            counter_virginica = 0
            counter_versicolor = 0
            for ob in clusters[i]:
                if ob.name == "Iris-setosa":
                    counter_setosa += 1
                if ob.name == "Iris-versicolor":
                    counter_versicolor += 1
                if ob.name == "Iris-virginica":
                    counter_virginica += 1

            print(f"Purity score {i+1}:"
                  f"\n\tIris-setosa: {counter_setosa/len(clusters[i])*100}"
                  f"\n\tIris-versicolor: {counter_versicolor/len(clusters[i])*100}"
                  f"\n\tIris-virginica: {counter_virginica/len(clusters[i])*100}")

    return clusters


def read(path, dest):

    with open(path, 'r') as f:
        for line in f:
            data = []
            name = ""
            list = line.split(',')
            for x in list:
                if list.index(x) == len(list)-1:
                    name = x.strip()
                else:
                    data.append(float(x.strip()))
            dest.append(observation(data, name))


if __name__ == '__main__':

    iris_set = []
    read("iris.data", iris_set)
    k = int(input("Provide k:\n"))
    kmeans(iris_set,k)
