import random


def run(layers: list[int], bias: float, learning_rate: float, momentum_rate: float, max_epoch: int, epsilon: float, file_path: str):
    two_weight_change = [[]]
    input_data = read_file(file_path)

    weights = create_weight(layers)
    node = forward_pass(weights, input_data, layers, bias)
    costs = calculate_cost(node, desire_output)
    gradient = find_grad(node, weights, costs, layers)

    two_weight_change.append(calculate_weight_change(node, gradient, two_weight_change[-1], weights, learning_rate, momentum_rate, layers, epoch))
    update_weight(two_weight_change[-1], weights, layers)
    two_weight_change.pop(0)
    epoch += 1

    return 0


def read_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    return lines


def format_data(lines: list[str]) -> list[list[list[float]]]:
    """
    format data to form
    [[[...input_data],[...desire_data]],[[...input_data],[...desire_data]],...,[[...input_data],[...desire_data]]]

    :rtype: list[list[list[float]]]
    :param lines: list of string each line from file
    :return: formatted data
    """
    data_list: list[list[list[float]]] = []
    for line in lines:
        temp = []
        raw_data = [float(y) for y in (line.split())]
        desire_output = [raw_data.pop()]
        temp.append(raw_data)
        temp.append(desire_output)
        data_list.append(temp)
    random.shuffle(data_list)
    return data_list


def create_node(layers: list) -> list[list[list[float]]]:
    """
    create list of lists of node in each layer

    :rtype: list
    :param layers: layers of this network
    :return: node (list of lists of node in each layer)
    """
    node = []
    temp2 = []
    for i in range(len(layers)):
        for j in range(layers[i]):
            temp2.append([0.0])
        node.append(temp2)
        temp2 = []
    return node


def create_weight(layers: list) -> list:
    """
    create lists of weight in each interval

    :rtype: list
    :param layers: layers of this network
    :return: weight
    """
    weight = []
    temp_k = []
    temp_j = []
    for i in range(len(layers) - 1):
        for k in range(layers[i + 1]):
            for j in range(layers[i]):
                temp_j.append(random.random())
            temp_k.append(temp_j)
            temp_j = []
        weight.append(temp_k)
        temp_k = []
    return weight


def find_sse_average(error: list[float]) -> float:
    """
    calculate sum square error average from list of errors

    :rtype: float
    :param error: list of errors
    :return: sse_average
    """
    sse_average = 0
    for n in error:
        sse_average += n * n
    sse_average = sse_average / len(error)
    return sse_average


def multiply_matrix(x: list, y: list) -> list:
    """
    multiply matrix x, y

    :rtype: list
    :param x: list of first matrix
    :param y: list of second matrix
    :return: result of x cross y
    """
    return [[sum(a * b for a, b in zip(x_row, y_col)) for y_col in zip(*y)] for x_row in x]


def forward_pass(weight: list, input_data: list, layers: list, bias: float) -> list[list[list[float]]]:
    """
    calculate node of network by matrix multiplication and return result matrix node

    :rtype: list[list[list[float]]]
    :param bias: bias of network
    :param weight:
    :param input_data:
    :param layers:
    :return: node
    """
    # insert input
    node = create_node(layers)
    node[0] = transpose(input_data)

    # calculate activation value
    for i in range(len(layers) - 1):
        node[i+1] = multiply_matrix(node[i], weight[i])
        node[i+1] = matrix_operation(node[i+1], activation_function)
        node[i+1] = two_matrix_operation(node[i+1], create_bias_vector(bias, len(node[i+1])), add_number)

    return node


def calculate_cost(node: list[list[list[float]]], desire_output: list[float]) -> list[float]:
    # find error
    error = []
    for i in range(len(node[-1])):
        error.append(desire_output[i] - node[-1][i][0])
    return error


def find_grad(node: list, weight: list, error: list, layers: list):
    """
    Calculate gradient in each node
    Output layer: error * diff_activation_function(multiply_matrix(node[-1],weight[-1]))
    hidden layer at i layer: diff_activation_function(multiply_matrix(node[len(node)-i-1],weight[len(weight)-i-1])) * multiply_matrix(grad[len(grad)-i],weight[len(weight)-i])

    :param layers: layers of this network
    :param node: list of lists of nodes in each layer
    :param weight: lists of weight in each interval
    :param error: error from forward_pass
    :return: list of lists of gradient in each layer
    """
    # Calculate gradient at output layer
    grad = create_node(layers)
    for i in range(len(error[-1])):
        grad[-1][i] = [error[-1][i] * diff_activation_function(multiply_matrix(node[-1], weight[-1])[-1][i])]

    # Ex. [1,2,1] i = {0}, {1} and j = {0,1}, {0}
    for i in range(len(layers) - 2):  # i = 0 -> N - 1(start at 0) - 2()
        for j in range(layers[len(layers) - i - 2]):  # j = 0 -> layer[N - i] -1
            current_layer_pos = len(layers) - 2 - i
            # fixed there is no bias here !!!
            grad[current_layer_pos] = two_matrix_operation(matrix_operation(multiply_matrix(weight[current_layer_pos - 1], node[current_layer_pos - 1]), diff_activation_function), multiply_matrix(weight[current_layer_pos - 1], node[current_layer_pos - 1]), multiply_number)
    return grad


def matrix_operation(input_matrix: list[list[float]], method) -> list:
    """
    apply method function to all element in matrix

    :rtype: list
    :param method:
    :param input_matrix: input vector or matrix
    :return: activation matrix

    """
    output_matrix = input_matrix
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[i])):
            output_matrix[i][j] = method(input_matrix[i][j])
    return output_matrix


def multiply_number(a: float, b: float) -> float:
    return a * b


def add_number(a: float, b: float) -> float:
    return a + b


def two_matrix_operation(input_matrix1: list[list[float]], input_matrix2: list[list[float]], method) -> list[list[float]]:
    """
    apply activation function to all element in matrix.

    :rtype: list[list[float]]
    :param input_matrix1: first input matrix
    :param input_matrix2: last input matrix
    :param method: input function
    :return: activation matrix
    """
    output_matrix = input_matrix1
    for i in range(len(input_matrix1)):
        for j in range(len(input_matrix1[i])):
            output_matrix[i][j] = method(input_matrix1[i][j], input_matrix2[i][j])
    return output_matrix


def create_bias_vector(bias: float, vector_size: int) -> list[list[float]]:
    """
    create bias vector

    :rtype: list[list[float]]
    :param bias: bias value
    :param vector_size: size of bias vector
    :return: vector of bias
    """
    bias_vector = []
    for i in range(vector_size):
        bias_vector.append([bias])
    return bias_vector


def activation_function(x: float) -> float:
    """
    calculate activation in this function use leaky relu

    :rtype: float
    :param x: input value in float
    :return: value when calculate activation
    """
    if x < 0:
        return 0.01
    else:
        return x


def diff_activation_function(x: float) -> float:
    """
    calculate diff activation in this function use leaky relu

    :rtype: float
    :param x: input value in float
    :return: value when calculate diff activation
    """
    if x < 0:
        return 0.01
    else:
        return 1


def transpose(input_matrix: list[list[float]]) -> list[list[float]]:
    """
    transpose input matrix

    :rtype: list
    :param input_matrix: in put vector or matrix
    :return: transpose matrix
    """
    return [[input_matrix[j][i] for j in range(len(input_matrix))] for i in range(len(input_matrix[0]))]


def calculate_weight_change(node: list[list[list[float]]], grad: list[list[list[float]]], last_weight_change: list[list[list[float]]], weight: list[list[list[float]]], learning_rate: float, momentum_rate: float, layers: list[float], epoch: float) -> list[list[list[float]]]:
    """
    calculate weight change of this network

    :rtype: list[list[list[float]]]
    :param node: matrix of activation value
    :param grad: gradient of  network
    :param last_weight_change: last weight change of current network
    :param weight: current weight
    :param learning_rate: learning rate speed range(0 - 1)
    :param momentum_rate: momentum rate speed range(0 - 1)
    :param layers: layers of current network
    :param epoch: round of epoch
    :return: matrix of weight change
    """
    weight_change = create_weight(layers)
    for i in range(len(weight)):
        for j in range(len(weight[i])):
            for k in range(len(weight[i][j])):
                if epoch == 0:
                    weight_change[i][j][k] = learning_rate * grad[i][j][0] * node[i][k][0]
                else:
                    weight_change[i][j][k] = momentum_rate * last_weight_change[i][j][k] + learning_rate * grad[i][j][0] * node[i][k][0]
    return weight_change


def update_weight(weight_change: list[list[list[float]]], weight: list[list[list[float]]], layers: list[float]) -> list[list[list[float]]]:
    new_weight = create_weight(layers)
    for i in range(len(weight)):
        for j in range(len(weight[i])):
            for k in range(len(weight[i][j])):
                new_weight[i][j][k] = weight[i][j][k] + weight_change[i][j][k]
    return new_weight


if __name__ == '__main__':
    run([8, 4, 1], 1, 0.1, 0.1, 2000, 0.005, './')
