import math
import collections
import matplotlib.pyplot as plt
import numpy as np

class observation:

    def __init__(self, data, name):
        self.data = data
        self.name = name
    def __str__(self):
        return 'data: '+str(self.data)+"name: "+self.name
    def __repr__(self):
        return str(self)

def kNN(ob,k,src):
    results = {}
    for x in src:
        sum = 0.0
        for y in range(0, len(x.data)):
            sum = sum + pow(x.data[y]-ob.data[y], 2)
        results[x] = math.sqrt(sum)
    results = sorted(results.items(), key=lambda x:x[1])
    results_names = [x[0].name for x in results[:k]]
    most_common = collections.Counter(results_names).most_common(1)[0][0]
    return most_common


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


def test(train_path,test_path,k):
    train_set = []
    test_set = []
    acc_counter = 0
    read(train_path, train_set)
    read(test_path, test_set)
    for ob in test_set:
        prediction = kNN(ob, k, train_set)
        if prediction == ob.name:
            acc_counter += 1
    return acc_counter/len(test_set)*100


if __name__ == '__main__':
    iris_accuracy = test("iris.data","iris.test.data",5)
    wdbc_accuracy = test("wdbc.data","wdbc.test.data",5)
    print("Accuracy of iris.test.data based on iris.data for k = 5: "+str(iris_accuracy)+"%\n"
            "Accuracy of wdbc.test.data based on wdbc.data for k = 5: "+str(wdbc_accuracy)+"%\n")
    choice = 1
    iris_set = []
    read("iris.data", iris_set)
    while choice != 0:
        choice = int(input("Type 0 to exit\nType 1 to continue\nType 2 to view relation graph for k and accuracy\n"))
        if choice == 1:
            user_in = input("Type 4 values, separated by coma, to be classified based on iris.data:\n")
            user_k = int(input("Provide k: "))
            user_data = [float(x) for x in user_in.split(',')]
            result_name = kNN(observation(user_data,""),user_k,iris_set)
            print("Data classified as "+result_name)
        elif choice == 2:
            user_k = int(input("Provide max k: "))
            data_points = []
            for i in range(1,user_k+1):
                data_points.append((i,test("iris.data","iris.test.data",i)))
            plt.plot(*zip(*data_points))
            plt.ylim(0,101)
            plt.yticks(np.arange(0,101,5))
            plt.xticks(np.arange(0,user_k+1,5))
            plt.xlabel("Value of k")
            plt.ylabel("Accuracy percentage")
            plt.show()
