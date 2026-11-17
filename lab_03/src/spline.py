"""3.2: Кубический сплайн с естественными граничными условиями.

Использует TridiagonalSolver (метод прогонки) из лабораторной работы №1
для решения трёхдиагональной системы относительно коэффициентов c_i.
"""
from typing import List, Tuple

from tridiag_solver import TridiagonalSolver


class CubicSpline:
    """Естественный кубический сплайн: S''(x0) = S''(xn) = 0.

    На участке [x_i, x_{i+1}] сплайн имеет вид:
        S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
    """

    def __init__(self, x: List[float], y: List[float]):
        if len(x) != len(y):
            raise ValueError("Количество узлов x и y должно совпадать")
        if len(x) < 3:
            raise ValueError("Для сплайна нужно минимум 3 узла")
        for i in range(len(x) - 1):
            if x[i + 1] <= x[i]:
                raise ValueError("Узлы x должны быть строго возрастающими")

        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.n = len(x)                 # число узлов
        self.segments = self.n - 1      # число участков

        # Коэффициенты: для каждого участка i — (a_i, b_i, c_i, d_i)
        # Для c нужен индекс до n-1 включительно (c[0] = c[n-1] = 0)
        self.a = [0.0] * self.segments
        self.b = [0.0] * self.segments
        self.c = [0.0] * self.n         # c[0] .. c[n-1], границы = 0
        self.d = [0.0] * self.segments

        self._solve()

    # ------------------------------------------------------------------
    # Решение системы для коэффициентов
    # ------------------------------------------------------------------
    def _solve(self):
        n = self.n
        seg = self.segments
        x = self.x
        y = self.y

        # Длины участков
        h = [x[i + 1] - x[i] for i in range(seg)]

        # Внутренние неизвестные c[1] .. c[n-2] — их (n - 2) штук
        m = n - 2
        if m > 0:
            # Диагональ: 2*(h[i] + h[i+1]) для i = 0..m-1
            diag = [2.0 * (h[i] + h[i + 1]) for i in range(m)]
            # Наддиагональ: h[i+1] для i = 0..m-2
            over = [h[i + 1] for i in range(m - 1)]
            # Поддиагональ: h[i+1] для i = 0..m-2
            sub = [h[i + 1] for i in range(m - 1)]
            # Правая часть
            # rhs[i] = 3 * ((y[i+2]-y[i+1])/h[i+1] - (y[i+1]-y[i])/h[i])
            rhs = [
                3.0 * ((y[i + 2] - y[i + 1]) / h[i + 1]
                       - (y[i + 1] - y[i]) / h[i])
                for i in range(m)
            ]

            solver = TridiagonalSolver(diag, over, sub, rhs)
            _, _, c_inner = solver.solve()

            # c_inner[0] = c[1], c_inner[1] = c[2], ..., c_inner[m-1] = c[m]
            for i in range(m):
                self.c[i + 1] = c_inner[i]

        # Восстанавливаем a, b, d
        for i in range(seg):
            self.a[i] = y[i]
            self.b[i] = (y[i + 1] - y[i]) / h[i] \
                        - h[i] * (2.0 * self.c[i] + self.c[i + 1]) / 3.0
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])

    # ------------------------------------------------------------------
    # Поиск интервала
    # ------------------------------------------------------------------
    def _find_interval(self, x_star: float) -> int:
        if x_star <= self.x[0]:
            return 0
        if x_star >= self.x[-1]:
            return self.segments - 1
        for i in range(self.segments):
            if self.x[i] <= x_star <= self.x[i + 1]:
                return i
        return self.segments - 1

    # ------------------------------------------------------------------
    # Вычисление значения
    # ------------------------------------------------------------------
    def value(self, x_star: float) -> Tuple[float, int]:
        """Возвращает (значение сплайна, номер интервала)."""
        idx = self._find_interval(x_star)
        t = x_star - self.x[idx]
        val = self.a[idx] + t * (self.b[idx] + t * (self.c[idx] + t * self.d[idx]))
        return val, idx

    def second_derivative(self, x_star: float) -> float:
        idx = self._find_interval(x_star)
        t = x_star - self.x[idx]
        return 2.0 * self.c[idx] + 6.0 * self.d[idx] * t

    # ------------------------------------------------------------------
    # Вывод
    # ------------------------------------------------------------------
    def coefficients_str(self) -> str:
        header = (f"{'i':>3} | {'a_i':>12} | {'b_i':>12} | "
                  f"{'c_i':>12} | {'d_i':>12} | Интервал")
        lines = [header, "-" * len(header)]
        for i in range(self.segments):
            interval = f"[{self.x[i]:.4f}, {self.x[i + 1]:.4f}]"
            lines.append(
                f"{i:>3} | {self.a[i]:>12.6f} | {self.b[i]:>12.6f} | "
                f"{self.c[i]:>12.6f} | {self.d[i]:>12.6f} | {interval}"
            )
        return "\n".join(lines)

    def plot_data(self, step: float = 0.02) -> Tuple[List[float], List[float]]:
        x_min, x_max = self.x[0], self.x[-1]
        count = int(round((x_max - x_min) / step))
        xs = [x_min + k * step for k in range(count + 1)]
        ys = [self.value(xv)[0] for xv in xs]
        return xs, ys

    # ------------------------------------------------------------------
    # Проверки (по требованию задания)
    # ------------------------------------------------------------------
    def check_nodes(self) -> List[float]:
        """Отклонения сплайна в узлах: |S(x_i) - y_i|. Должны быть ~0."""
        return [abs(self.value(x)[0] - y) for x, y in zip(self.x, self.y)]

    def boundary_second_derivatives(self) -> Tuple[float, float]:
        """Вторая производная на концах. Для естественного сплайна — ~0."""
        return (self.second_derivative(self.x[0]),
                self.second_derivative(self.x[-1]))
