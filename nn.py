from matrix import Matrix
import random
import math
import json

class LinearLayer:
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        self.weights = Matrix(output_features, input_features)
        self.biases = Matrix(output_features, 1)

        self.weights.randomize()
        self.biases.randomize()

    def __call__(self, x):
        return Matrix.add(Matrix.multiply(self.weights, x), self.biases)


def RELU(matrix):
    return Matrix.map_static(matrix, lambda x: max(0, x))

def dRELU(matrix):
    return Matrix.map_static(matrix, lambda x: 1 if x > 0 else 0)

def softmax(matrix):

    # Apply softmax to each row of the matrix (assuming matrix is a 2D list or similar structure)
    # First, compute the exponentials for each element in the matrix
    sum = 0
    max_exp_value = 20
    exp_matrix = Matrix.map_static(matrix, lambda x: math.exp(min(x, max_exp_value)))
    
    # Now, for each row, we compute the sum of exponentials (to normalize)
    for row in exp_matrix.data:
        for col in row:
            sum += col
    
    # Normalize each element by dividing by the sum of the row
    softmax_matrix = Matrix.map_static(exp_matrix, lambda x: x/sum)
    
    return softmax_matrix

class NeuralNetwork:
    
    def __init__(self, input_features, output_features):
        self.l1 = LinearLayer(input_features, 10)
        self.l2 = LinearLayer(10, output_features)

    def forward(self, x):
        x = self.l1(x)
        x = RELU(x)

        x_h = Matrix.clone(x)

        x = self.l2(x)
        x = softmax(x)

        return x, x_h
    
    def backward(self, lr, X, target):

    # The second layer
        # Get the predictions for both layers.
        pred, x_h = self.forward(X)

        # Computer dLoss with respect to Z of the 2nd layer
        dZ2 = Matrix.clone(pred)
        # The formula is dZ2 = Y-1[j=c]
        for i, row in enumerate(target.data):
            for j, col in enumerate(row):
                if col == 1:
                    dZ2.data[i][j] -= 1
                    break

        # Computer dLoss with respect to W of the 2nd layer
        dW2 = Matrix.multiply(dZ2, Matrix.transpose(x_h))

        # Computer dLoss with respect to b of the 2nd layer
        # It's the same as dZ2
        db2 = Matrix.clone(dZ2)

        # Computer dLoss with respect to the output of the hidden layer
        dH = Matrix.multiply(Matrix.transpose(self.l2.weights), dZ2)

    # The first layer
        # Computer dLoss with respect to Z of the 1st layer
        Z1 = self.l1(X)
        dZ1 = Matrix.multiply_elementwise(dH, dRELU(Z1))

        # Computer dLoss with respect to W of the 1st layer
        dW1 = Matrix.multiply(dZ1, Matrix.transpose(X))

        # Computer dLoss with respect to b of the 1st layer
        db1 = Matrix.clone(dZ1)

    # Scale down
        dW2 = Matrix.multiply_elementwise(dW2, lr)
        db2 = Matrix.multiply_elementwise(db2, lr)
        dW1 = Matrix.multiply_elementwise(dW1, lr)
        db1 = Matrix.multiply_elementwise(db1, lr)

    # Fun Part: Update all the Parameters
        self.l2.weights = self.l2.weights.subtract(dW2)
        self.l2.biases = self.l2.biases.subtract(db2)
        self.l1.weights = self.l1.weights.subtract(dW1)
        self.l1.biases = self.l1.biases.subtract(db1)

# def nll_loss(pred, target): 
#     # Both are 3 * 1 Matrix or a Column vector.

#     index = None

#     for i, row in enumerate(target.data):
#         for col in row:
#             if col == 1:
#                 index = i
#                 break

#     return -math.log(pred.data[index][0])

def nll_loss(pred, target): 
    # Both are 3 * 1 Matrix or a Column vector.

    index = None

    for i, row in enumerate(target.data):
        for col in row:
            if col == 1:
                index = i
                break

    epsilon = 1e-10
    # Using max to avoid log(0)
    value = pred.data[index][0]
    result = -math.log(max(value, epsilon))

    return result


# def load_model(nn):
#     with open('nn_2.json', 'r') as f:
#         parameters = json.load(f)

#     # Assuming weights and biases are numpy arrays or similar
#     nn.l1.weights.data = parameters['weights_1']
#     nn.l2.weights.data = parameters['weights_2']
#     nn.l1.biases.data = parameters['biases_1']
#     nn.l2.biases.data = parameters['biases_2']

#     print("Model loaded successfully.")

# def save_model(nn, name):
#     parameters = {
#         'weights_1': nn.l1.weights.data,
#         'weights_2': nn.l2.weights.data,
#         'biases_1': nn.l1.biases.data,
#         'biases_2': nn.l2.biases.data
#     }

#     with open(name, 'w') as f:
#         json.dump(parameters, f)
#     print("Model saved successfully.")

# total_data = []

# for j in range(2):
#     print(j+1)
#     with open(f'data_{j+1}.json', 'r') as f:
#         datas = json.load(f)

#     for i in range(len(datas)):
#         data = datas[i][str(i+1)]
#         for dat in data:
#             total_data.append(dat)

# random.shuffle(total_data)

# my_nn = NeuralNetwork(5, 3)
# load_model(my_nn)

# print(len(total_data))

# epochs = 300

# for epoch in range(epochs):
#     total_loss = 0
#     for i in range(len(total_data)):

#         lr = 0.00005

#         if i % 150 == 0:
#             lr = 0.00001

#         data = total_data[i]
#         X = Matrix.from_array(data[0])
#         label = Matrix.from_array(data[1])

#         pred, _ = my_nn.forward(X)
#         total_loss += nll_loss(pred, label)
#         my_nn.backward(lr, X, label)

#     print(f'Total loss for epoch {epoch}: {total_loss/len(total_data)}')

# save_model(my_nn, 'nn_2.json')


# x = Matrix(5, 1)
# x.data = [
#     [65.04545811272106],
#     [86.20341529541477],
#     [209.37293842268633],
#     [317.3993935091874],
#     [81.9564361223388]
# ]

# label = Matrix(3, 1)
# label.data = [[1], [0], [0]]
# pred, _ = my_nn.forward(x)
# print('Prediction: ')
# pred.print()

# for i in range(1):
#     # print('b2 -------------------------')
#     # my_nn.l2.biases.print()
#     pred, _ = my_nn.forward(x)
#     if (i % 1 == 0):
#         print('loss: ', nll_loss(pred, label))
#     my_nn.backward(100, x, label)

# pred, _ = my_nn.forward(x)
# print('Prediction: ')
# pred.print()

