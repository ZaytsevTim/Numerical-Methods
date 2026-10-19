class IterationSolver:
    def __init__(self, A, b, eps):
        """
        Конструктор класса.
        A: исходная квадратная матрица (список списков float)
        b: вектор свободных членов (список float)
        eps: заданная точность останова (float)
        """
        self.A = [[float(x) for x in row] for row in A]
        self.b = [float(x) for x in b]
        self.eps = float(eps)
        self.N = len(b)
        self.diag_eps = 1e-15  # <-- порог «нулевой диагонали»

    def check_diagonal_dominance(self):
        """
        Проверка условий сходимости (достаточное условие диагонального преобладания).
        """
        has_strict = False
        for i in range(self.N):
            diag = abs(self.A[i][i])
            row_sum = sum(abs(self.A[i][j]) for j in range(self.N) if i != j)
            if diag < row_sum:
                return False
            if diag > row_sum:
                has_strict = True
        return has_strict

    def _check_no_zero_diagonal(self):
        """
        Проверяет, что на главной диагонали нет нулей (иначе деление невозможно).
        Вызывается один раз до старта итераций.
        """
        for i in range(self.N):
            if abs(self.A[i][i]) < self.diag_eps:
                raise ValueError(
                    f"Нулевой диагональный элемент A[{i}][{i}] = {self.A[i][i]}. "
                    f"Метод простых итераций/Зейделя в данной форме не применим. "
                    f"Переставьте строки/столбцы так, чтобы диагональ была ненулевой."
                )

    def _run_engine(self, method_type="seidel", max_iterations=10000):
        """
        Универсальный итерационный движок для МПИ (Якоби) и Зейделя (DRY).
        """
        # 1. Проверяем, что деление на диагональ вообще возможно
        self._check_no_zero_diagonal()

        # 2. Предупреждаем, если сходимость не гарантирована по теории
        if not self.check_diagonal_dominance():
            print("[Предупреждение] Диагональное преобладание не выполнено! "
                  "Сходимость итераций не гарантирована.")

        x = [0.0 for _ in range(self.N)]
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            x_read = list(x)  # Честный снимок вектора ДО начала текущего раунда

            for i in range(self.N):
                current_sum = self.b[i]
                for j in range(self.N):
                    if i != j:
                        # Якоби читает строго старый снимок x_read, Зейдель — живой x
                        if method_type == "jacobi":
                            current_sum -= self.A[i][j] * x_read[j]
                        else:
                            current_sum -= self.A[i][j] * x[j]

                x[i] = current_sum / self.A[i][i]

            # Критерий сходимости: max|x_new − x_old|
            error = max(abs(x[k] - x_read[k]) for k in range(self.N))

            if error < self.eps:
                return x, iterations

        raise ValueError(f"Метод {method_type} не сошёлся за {max_iterations} итераций!")

    def solve_jacobi(self):
        """ Публичный метод для запуска метода простых итераций (Якоби) """
        return self._run_engine(method_type="jacobi")

    def solve_seidel(self):
        """ Публичный метод для запуска метода Зейделя """
        return self._run_engine(method_type="seidel")