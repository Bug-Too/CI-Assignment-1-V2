import random


def run(layers, bias, learning_rate, momentum_rate, max_epoch, epsilon, file_path):
    return 0


def read_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    return lines


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


def forward_pass(weight: list, input_data: list, layers: list) -> list[list[list[float]]]:
    """
    calculate node of network by matrix multiplication and return result matrix node

    :rtype: list[list[list[float]]]
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

    return node


def calculate_cost(node: list[list[list[float]]], desire_output: list[float]) -> list[float]:
    # find error
    error = []
    for i in range(len(node[-1])):
        error.append(desire_output[i] - node[-1][i][0])
    return error


def find_grad(node: list, weight: list, error: list, layers: list, bias: float):
    """
    Calculate gradient in each node
    Output layer: error * diff_activation_function(multiply_matrix(node[-1],weight[-1]))
    hidden layer at i layer: diff_activation_function(multiply_matrix(node[len(node)-i-1],weight[len(weight)-i-1])) * multiply_matrix(grad[len(grad)-i],weight[len(weight)-i])

    :param layers: layers of this network
    :param node: list of lists of nodes in each layer
    :param weight: lists of weight in each interval
    :param error: error from forward_pass
    :param bias: bias of network
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

    :param method:
    :param input_matrix: input vector or matrix
    :return: activation matrix
    :rtype: list
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

    :param input_matrix1: first input matrix
    :param input_matrix2: last input matrix
    :param method: input function
    :return: activation matrix
    :rtype: list[list[float]]
    """
    output_matrix = input_matrix1
    for i in range(len(input_matrix1)):
        for j in range(len(input_matrix1[i])):
            output_matrix[i][j] = method(input_matrix1[i][j], input_matrix2[i][j])
    return output_matrix


def create_bias_vector(bias: float, vector_size: int) -> list[list[float]]:
    """
    create bias vector

    :param bias: bias value
    :param vector_size: size of bias vector
    :return: vector of bias
    :rtype: list[list[float]]
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


if __name__ == '__main__':
    run([8, 4, 1], 1, 0.1, 0.1, 2000, 0.005, './')
