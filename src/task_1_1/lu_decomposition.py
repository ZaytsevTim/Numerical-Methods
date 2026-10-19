from matrix_utils import matrix_multiply

class LUPDecomposition:
    def __init__(self, A, eps=1e-9):
        """ Конструктор: выделяет память под L, U, P и запускает разложение """
        self.N = len(A)
        self.eps = float(eps)  # <-- сохраняем eps как атрибут
        self.U = [[float(element) for element in row] for row in A]
        self.L = [[1.0 if i == j else 0.0 for j in range(self.N)] for i in range(self.N)]
        self.P = [[1.0 if i == j else 0.0 for j in range(self.N)] for i in range(self.N)]
        self.swaps_count = 0
        self._decompose()

    def _decompose(self):
        """ LUP-разложение с выбором опорного элемента по столбцу с учетом self.eps """
        eps = self.eps  # <-- читаем из атрибута
        for i in range(self.N):
            max_val = abs(self.U[i][i])
            pivot_row = i

            for r in range(i + 1, self.N):
                val = abs(self.U[r][i])
                if val - max_val > eps:
                    max_val = val
                    pivot_row = r

            if max_val < eps:
                raise ValueError("Матрица вырожденная, LU-разложение невозможно!")

            if pivot_row != i:
                self.U[i], self.U[pivot_row] = self.U[pivot_row], self.U[i]
                self.P[i], self.P[pivot_row] = self.P[pivot_row], self.P[i]
                for c in range(i):
                    self.L[i][c], self.L[pivot_row][c] = self.L[pivot_row][c], self.L[i][c]
                self.swaps_count += 1

            for r in range(i + 1, self.N):
                if abs(self.U[r][i]) < eps:
                    self.L[r][i] = 0.0
                    self.U[r][i] = 0.0
                    continue

                factor = self.U[r][i] / self.U[i][i]
                self.L[r][i] = factor
                for c in range(i, self.N):
                    self.U[r][c] -= factor * self.U[i][c]

    def _forward(self, pb):
        """ Прямой ход: решает L * y = P * b """
        y = [0.0 for _ in range(self.N)]
        for i in range(self.N):
            sum_ly = pb[i]
            for j in range(i):
                sum_ly -= self.L[i][j] * y[j]
            y[i] = sum_ly
        return y

    def _backward(self, y):
        """ Обратный ход: решает U * x = y """
        x = [0.0 for _ in range(self.N)]
        for i in range(self.N - 1, -1, -1):
            sum_ux = y[i]
            for j in range(i + 1, self.N):
                sum_ux -= self.U[i][j] * x[j]
            diag = self.U[i][i]
            if abs(diag) < self.eps:  # <-- было 1e-9, теперь self.eps
                raise ValueError(f"lu: нулевой диагональный элемент U[{i}][{i}]")
            x[i] = sum_ux / diag
        return x

    def solve(self, b):
        """ Решает систему Ax = b через последовательный вызов forward/backward """
        b_column = [[element] for element in b]
        pb_matrix = matrix_multiply(self.P, b_column)
        pb = [row[0] for row in pb_matrix]

        y = self._forward(pb)
        x = self._backward(y)
        return x

    def get_determinant(self):
        """ Считает определитель матрицы с учетом знака перестановок """
        det = 1.0
        for i in range(self.N):
            det *= self.U[i][i]
        sign = 1.0 if self.swaps_count % 2 == 0 else -1.0
        return sign * det

    def get_inverse(self):
        """ Находит обратную матрицу A^-1 по столбцам """
        inv = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
        for j in range(self.N):
            e = [1.0 if i == j else 0.0 for i in range(self.N)]
            x_column = self.solve(e)
            for i in range(self.N):
                inv[i][j] = x_column[i]
        return inv