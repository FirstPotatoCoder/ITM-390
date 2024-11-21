import random
import struct

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0 for _ in range(cols)] for _ in range(rows)]

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = round(random.uniform(-1, 1), 6)

    @staticmethod
    def add(self, n):
        if isinstance(n, Matrix):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + n.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + n
            return result
  
    def subtract(self, n):
        if isinstance(n, Matrix):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - n.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - n
            return result

    # @staticmethod
    # def subtract(a, b):
    #     result = Matrix(a.rows, a.cols)
    #     for i in range(a.rows):
    #         for j in range(a.cols):
    #             result.data[i][j] = a.data[i][j] - b.data[i][j]
    #     return result

    @staticmethod
    def transpose(matrix):
        result = Matrix(matrix.cols, matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[j][i] = matrix.data[i][j]
        return result

    @staticmethod
    def multiply(a, b):
        if a.cols != b.rows:
            raise ValueError('Invalid inputs for multiplication')
        result = Matrix(a.rows, b.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                sum = 0
                for k in range(a.cols):
                    sum += a.data[i][k] * b.data[k][j]
                result.data[i][j] = sum
        return result

    # def multiply_elementwise(self, n):
    #     if isinstance(n, Matrix):
    #         for i in range(self.rows):
    #             for j in range(self.cols):
    #                 self.data[i][j] *= n.data[i][j]
    #     else:
    #         for i in range(self.rows):
    #             for j in range(self.cols):
    #                 self.data[i][j] *= n

    @staticmethod
    def multiply_elementwise(m1, m2):
        # Check if both are matrices (2D arrays)
        if isinstance(m1, Matrix) and isinstance(m2, Matrix):
            # Ensure both matrices have the same dimensions
            if m1.rows != m2.rows or m1.cols != m2.cols:
                raise ValueError("Matrices must have the same dimensions for elementwise multiplication.")
            
            # Create a new Matrix to store the result
            result_matrix = Matrix(len(m1.data), len(m1.data[0]))
            for i in range(len(m1.data)):
                for j in range(len(m1.data[0])):
                    result_matrix.data[i][j] = m1.data[i][j] * m2.data[i][j]

            return result_matrix 
        elif isinstance(m1, Matrix):
            # Scalar multiplication
            result_matrix = Matrix(m1.rows, m1.cols)
            for i in range(m1.rows):
                for j in range(m1.cols):
                    result_matrix.data[i][j] = m1.data[i][j] * m2

            return result_matrix
        else:
            raise ValueError("First argument must be a Matrix.")

    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j])

    @staticmethod
    def map_static(matrix, func):
        result = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[i][j] = func(matrix.data[i][j])
        return result

    @staticmethod
    def clone(matrix):
        result = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            result.data[i] = matrix.data[i].copy()
        return result

    @staticmethod
    def from_array(input_array):
        result = Matrix(len(input_array), 1)
        for i in range(len(input_array)):
            result.data[i][0] = input_array[i]
        return result

    @staticmethod
    def special_from_array(input_array, a, b):
        result = Matrix(a, b)
        k = 0
        for i in range(a):
            for j in range(b):
                result.data[i][j] = input_array[k]
                k += 1
        return result

    def to_array(self):
        return [self.data[i][j] for i in range(self.rows) for j in range(self.cols)]

    def print(self):
        for row in self.data:
            print(row)

# # Example usage
# if __name__ == "__main__":
#     weights = Matrix(10, 5)
#     weights.randomize()
#     weights.print()

#     print('----------------------')

#     biases = Matrix(10, 1)
#     biases.randomize()
#     biases.print()

#     print('----------------------')

#     x = Matrix(5, 1)
#     x.randomize()
#     x.print()

#     print('----------------------')

#     Matrix.add(Matrix.multiply(weights, x), biases)
