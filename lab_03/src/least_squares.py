from typing import List, Tuple


class LeastSquaresPolynomial:
    """Многочлен метода наименьших квадратов заданной степени."""

    def __init__(self, x: List[float], y: List[float], degree: int):
        if len(x) != len(y):
            raise ValueError("Количество узлов x и y должно совпадать")
        if len(x) < degree + 2:
            raise ValueError(f"Нужно минимум {degree + 2} точек для многочлена степени {degree}")
        self.x = [float(xi) for xi in x]
        self.y = [float(yi) for yi in y]
        self.n = len(x)
        self.degree = degree
        self.coeffs = self._solve_normal_equations()

    def _moments(self) -> List[float]:
        """Вычисляет моменты S_k = sum(x_i^k)."""
        moments = [0.0] * (2 * self.degree + 1)
        for k in range(2 * self.degree + 1):
            moments[k] = sum(xi ** k for xi in self.x)
        return moments

    def _rhs(self) -> List[float]:
        """Вычисляет правые части системы (S_ky)."""
        rhs = [0.0] * (self.degree + 1)
        for k in range(self.degree + 1):
            rhs[k] = sum((xi ** k) * yi for xi, yi in zip(self.x, self.y))
        return rhs

    def _solve_normal_equations(self) -> List[float]:
        """Решает нормальную систему уравнений методом Гаусса."""
        m = self.degree + 1
        moments = self._moments()
        rhs = self._rhs()

        # Строим матрицу системы
        mat = [[0.0] * (m + 1) for _ in range(m)]
        for i in range(m):
            for j in range(m):
                mat[i][j] = moments[i + j]
            mat[i][m] = rhs[i]

        # Метод Гаусса с выбором ведущего элемента
        for col in range(m):
            # Поиск ведущего элемента
            max_row = col
            for row in range(col + 1, m):
                if abs(mat[row][col]) > abs(mat[max_row][col]):
                    max_row = row
            mat[col], mat[max_row] = mat[max_row], mat[col]

            if abs(mat[col][col]) < 1e-12:
                raise ValueError("Система вырождена")

            # Исключаем переменную
            for row in range(col + 1, m):
                factor = mat[row][col] / mat[col][col]
                for j in range(col, m + 1):
                    mat[row][j] -= factor * mat[col][j]

        # Обратная подстановка
        coeffs = [0.0] * m
        for i in range(m - 1, -1, -1):
            coeffs[i] = mat[i][m]
            for j in range(i + 1, m):
                coeffs[i] -= mat[i][j] * coeffs[j]
            coeffs[i] /= mat[i][i]
        return coeffs

    def value(self, x_star: float) -> float:
        """Вычисляет значение многочлена в точке x_star (по схеме Горнера)."""
        result = 0.0
        for c in reversed(self.coeffs):
            result = result * x_star + c
        return result

    def polynomial_str(self) -> str:
        """Возвращает строковое представление многочлена."""
        terms = []
        for i, c in enumerate(reversed(self.coeffs)):
            if abs(c) < 1e-10:
                continue
            power = self.degree - i
            if power == 0:
                term = f"{c:.4g}"
            elif power == 1:
                term = f"{c:.4g} * x" if abs(c) != 1.0 else "x"
            else:
                coef = f"{c:.4g}" if abs(c) != 1.0 else ""
                term = f"{coef} * x^{power}" if coef else f"x^{power}"
            terms.append((power, term))
        # Собираем строку
        terms.sort(reverse=True)
        result = ""
        for power, term in terms:
            if result:
                result += " + " if term[0] != "-" else " - "
                if term[0] == "-":
                    result += term[1:]
                else:
                    result += term
            else:
                result = term
        return result

    def mse(self) -> float:
        """Вычисляет среднеквадратичную ошибку на исходных данных."""
        sum_sq = 0.0
        for xi, yi in zip(self.x, self.y):
            sum_sq += (self.value(xi) - yi) ** 2
        return sum_sq / self.n

    def plot_data(self, step: float = 0.02) -> Tuple[List[float], List[float]]:
        """Генерирует точки для графика."""
        xs, ys = [], []
        x_min, x_max = min(self.x), max(self.x)
        for xi in range(int(x_min / step), int(x_max / step) + 1):
            x_val = xi * step
            xs.append(x_val)
            ys.append(self.value(x_val))
        return xs, ys


def create_least_squares_results(x: List[float], y: List[float]):
    """Создаёт результаты для многочленов 1, 2, 3 степени."""
    p1 = LeastSquaresPolynomial(x, y, 1)
    p2 = LeastSquaresPolynomial(x, y, 2)
    p3 = LeastSquaresPolynomial(x, y, 3)
    return {
        "p1": p1, "mse1": p1.mse(),
        "p2": p2, "mse2": p2.mse(),
        "p3": p3, "mse3": p3.mse(),
    }
