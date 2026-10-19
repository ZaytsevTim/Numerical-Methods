import math
import numpy as np

from matrix_utils import dot


class QRDecompositionSolver:
    def __init__(self, A, eps):
        """
        Конструктор класса.
        A: исходная квадратная матрица (список списков float)
        eps: заданная точность останова для поддиагональных элементов (float)
        """
        self.eps = float(eps)
        self.N = len(A)
        self.A = [[float(x) for x in row] for row in A]
        self.iterations = 0

    def householder_qr(self, matrix):
        """
        Публичный метод: раскладывает матрицу на Q и R методом Хаусхолдера.
        Возвращает ортогональную матрицу Q и верхнюю треугольную R такие, что A = Q * R.
        """
        N = len(matrix)
        R = [row[:] for row in matrix]
        # Q стартует как единичная матрица E размера N x N
        Q = [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]

        # Идём по столбцам, кроме последнего
        for k in range(N - 1):
            # Вырезаем нижний хвост k-го столбца, начиная с R[k][k]
            v = [R[i][k] for i in range(k, N)]

            # Считаем норму хвоста как sqrt(v · v)
            norm_v = math.sqrt(dot(v, v))
            if norm_v < 1e-15:
                continue

            # Целевой вектор: вся длина уходит в R[k][k]
            sign = 1.0 if v[0] >= 0 else -1.0
            target = [0.0] * len(v)
            target[0] = sign * norm_v

            # Вектор зеркала w = v - target
            w = [v[i] - target[i] for i in range(len(v))]

            # Нормируем w, чтобы матрица отражения была H = E - 2 * w * w^T
            norm_w = math.sqrt(dot(w, w))
            if norm_w < 1e-15:
                continue
            w = [x / norm_w for x in w]

            # Применяем H слева к R: R = H * R
            # H * R = R - 2 * w * (w^T * R), где w^T * R дает вектор по столбцам
            for j in range(k, N):
                # Хвост j-го столбца R, начиная с k-й строки
                col_j = [R[m][j] for m in range(k, N)]
                # Скалярное произведение w на этот столбец
                d = dot(w, col_j)
                for i in range(k, N):
                    R[i][j] -= 2.0 * w[i - k] * d

            # Накапливаем Q: Q = Q * H (H симметрична, поэтому H = H^T)
            # Q * H = Q - 2 * (Q * w) * w^T, где Q * w дает вектор по строкам
            for i in range(N):
                # Скалярное произведение хвоста i-й строки Q на w
                d = dot(Q[i][k:], w)
                for j in range(k, N):
                    Q[i][j] -= 2.0 * d * w[j - k]

        return Q, R

    def solve(self):
        """
        Выполняет QR-итерации A <- R * Q до сходимости поддиагональных элементов.
        Возвращает список собственных значений и число итераций.
        """
        A = self.A
        N = self.N
        self.iterations = 0
        # Лимит итераций: для матриц с комплексной парой QR без сдвигов
        # принципиально не зануляет поддиагональ, поэтому лимит небольшой.
        max_iter = 1000

        while self.iterations < max_iter:
            # Критерий остановки: все поддиагональные элементы меньше eps
            converged = True
            for i in range(N - 1):
                if abs(A[i + 1][i]) >= self.eps:
                    converged = False
                    break
            if converged:
                break

            # Один шаг QR: A = Q * R, затем A_new = R * Q
            Q, R = self.householder_qr(A)
            # A_new[i][j] = R_i · Q^j  (строка на столбец)
            A = [[dot(R[i], [Q[k][j] for k in range(N)])
                  for j in range(N)] for i in range(N)]
            self.iterations += 1

        # Информационное сообщение, если за max_iter так и не сошлось
        if self.iterations >= max_iter:
            print(f"[Информация] QR-алгоритм остановлен по лимиту итераций "
                  f"({max_iter}). Возможно, есть комплексная пара собственных значений.")

        self.A = A
        return self._extract_eigenvalues()

    def _extract_eigenvalues(self):
        """
        Извлекает собственные значения из почти треугольной матрицы.
        Для блока 2x2 (комплексно-сопряжённая пара) используется numpy.linalg.eigvals,
        поскольку блок не зануляется вещественными отражениями.
        """
        eigenvalues = []
        i = 0
        while i < self.N:
            # Вещественное собственное значение на диагонали
            if i == self.N - 1 or abs(self.A[i + 1][i]) < self.eps:
                eigenvalues.append(complex(self.A[i][i], 0.0))
                i += 1
            else:
                # Блок 2x2, который не зануляется — здесь numpy разрешён по ТЗ
                block = [
                    [self.A[i][i],     self.A[i][i + 1]],
                    [self.A[i + 1][i], self.A[i + 1][i + 1]],
                ]
                roots = np.linalg.eigvals(np.array(block, dtype=float))
                eigenvalues.append(complex(roots[0]))
                eigenvalues.append(complex(roots[1]))
                i += 2
        return eigenvalues, self.iterations