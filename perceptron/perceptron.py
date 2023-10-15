import random

class observation:

    def __init__(self, data, name):
        self.data = data
        self.name = name
    def __str__(self):
        return 'data: '+str(self.data)+"name: "+self.name
    def __repr__(self):
        return str(self)

class perceptron:

    def __init__(self,target,learning_rate,epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.target = target
        self.theta = 1
        self.weights = None

    def activation(self,net):
        return 1 if net >= 0.0 else 0

    def train(self,data):
        self.weights = [random.uniform(0.0,1.0) for _ in range(len(data[0].data))]
        y = 0
        net = 0
        for _ in range(self.epochs):
            for ob in data:
                if ob.name == self.target:
                    y = 1
                else:
                    y = 0

                net = sum(ob.data[i]*self.weights[i] for i in range(len(ob.data))) - self.theta
                predicted = self.activation(net)
                update = self.learning_rate*(y - predicted)
                self.weights = [self.weights[i]+update*ob.data[i] for i in range(len(self.weights))]
                self.theta = self.theta - update

    def predict(self,data):
        net = sum(data[i]*self.weights[i] for i in range(len(data))) - self.theta
        predicted = self.activation(net)
        return self.target if predicted == 1 else "not "+self.target

def read(path):

    result = []
    with open(path,'r', encoding='utf-8') as f:
        for line in f:
            data = []
            name = ""
            list = line.split(',')
            for x in list:
                if list.index(x) == len(list)-1:
                    name = x.strip()
                else:
                    data.append(float(x.strip()))
            result.append(observation(data, name))
    return result

if __name__ == '__main__':

    train_set = read("perceptron.data")
    test_set = read("perceptron.test.data")

    perceptron = perceptron("Iris-versicolor",0.01,1000)
    perceptron.train(train_set)
    predictions = {}
    errors = 0
    for ob in test_set:
        prediction = perceptron.predict(ob.data)
        if prediction != ob.name and perceptron.target == ob.name:
            errors += 1
    accuracy = (len(test_set) - errors)/len(test_set) * 100
    print(f"accuracy: {accuracy}%")
    choice = 1
    while choice != 0:
        user_in = input("Provide 4 values, separated by coma, to be classified:\n")
        user_data = [float(x) for x in user_in.split(',')]
        result_name = perceptron.predict(user_data)
        print(f"Data classified as {result_name}")
        choice = int(input("Type 0 to exit\nType 1 to continue\n"))

