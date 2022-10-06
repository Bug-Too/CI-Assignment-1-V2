from unittest import TestCase

import main


class Test(TestCase):
    def test_multiply_matrix(self):
        self.assertEqual(main.multiply_matrix([[1, 1], [1, 1]], [[0.5, 1], [1, 0.5]]), [[1.5, 1.5], [1.5, 1.5]],
                         'Testcase:[[1, 1], [1, 1]], [[0.5, 1], [1, 0.5]] -> [[1.5, 1.5], [1.5, 1.5]]')
        self.assertEqual(main.multiply_matrix([[1]], [[1]]), [[1]], 'Testcase:[[1]], [[1]] -> [[1]]')
        self.assertEqual(main.multiply_matrix([[1, 1, 0], [1, 1, 0]], [[0.5, 1], [1, 0.5], [1, 1]]),
                         [[1.5, 1.5], [1.5, 1.5]],
                         '[[1, 1, 0], [1, 1, 0]], [[0.5, 1], [1, 0.5], [1, 1]]-> [[1.5, 1.5], [1.5, 1.5]]')

    def test_find_sse_average(self):
        self.assertEqual(main.find_sse_average([1, 1, 0, 1]), 0.75, 'Testcase:[1, 1, 0, 1] -> 0.75')
        self.assertEqual(main.find_sse_average([1, 1, 1]), 1, 'Testcase: [1, 1, 1] -> 1')
        self.assertEqual(main.find_sse_average([-1, 1, 0, 0]), 0.5, 'Testcase: [-1, 1, 0, 0] -> 0.5')

    def test_create_weight(self):
        print('result: ', main.create_weight([1, 1]), 'expected: [[[*]]]')
        print('result: ', main.create_weight([1, 2]), 'expected: [[[*],[*]]')
        print('result: ', main.create_weight([1, 2, 1]), 'expected: [[[*],[*]],[[*],[*]]]')
        print(main.create_weight([1, 2, 2, 1]))

    def test_loop(self):
        layers = [1, 2, 2, 1]
        # Ex. [1,2,1] i = {0}, {1} and j = {0,1}, {0}
        for i in range(len(layers) - 2):  # i = 0 -> N - 1(start at 0) - 2()
            for j in range(layers[len(layers) - i - 2]):  # j = 0 -> layer[N - i] -1
                print(len(layers) - 2 - i, j)

    def test_transpose(self):
        self.assertEqual(main.transpose([[1]]), [[1]])
        self.assertEqual(main.transpose([[1], [1]]), [[1, 1]])
        