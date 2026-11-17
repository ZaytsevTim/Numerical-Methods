"""3.1: Интерполяционные многочлены Лагранжа и Ньютона."""
from typing import List, Tuple


# ---------- Вспомогательные функции для работы с многочленами ----------
# Многочлен = список коэффициентов от младшей степени к старшей.

def poly_mul_shift(a: List[float], root: float) -> List[float]:
    """Умножает многочлен a на (x - root)."""
    r = [0.0] * (len(a) + 1)
    for i, c in enumerate(a):
        r[i + 1] += c
        r[i] += -root * c
    return r


def poly_eval(coef: List[float], x: float) -> float:
    """Схема Горнера."""
    res = 0.0
    for c in reversed(coef):
        res = res * x + c
    return res


def poly_str(coef: List[float], var: str = "x") -> str:
    """Строковое представление многочлена в явном виде."""
    parts = []
    for i in range(len(coef) - 1, -1, -1):
        c = coef[i]
        if abs(c) < 1e-12:
            continue
        if i == 0:
            parts.append(f"{c:+.6g}")
        elif i == 1:
            parts.append(f"{c:+.6g}*{var}")
        else:
            parts.append(f"{c:+.6g}*{var}^{i}")
    if not parts:
        return "0"
    result = parts[0].lstrip("+")
    for p in parts[1:]:
        result += " " + p
    return result


# ---------- Лагранж ----------
class LagrangeInterpolator:
    """Многочлен Лагранжа (в явном виде — через коэффициенты)."""

    def __init__(self, x: List[float], y: List[float]):
        if len(x) != len(y):
            raise ValueError("Количество x и y должно совпадать")
        if len(x) < 2:
            raise ValueError("Нужно минимум 2 узла")
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.n = len(x)
        self.coef = self._build_coefficients()

    def _build_coefficients(self) -> List[float]:
        n = self.n
        coef = [0.0] * n
        for i in range(n):
            basis = [1.0]
            scale = 1.0
            for j in range(n):
                if j != i:
                    basis = poly_mul_shift(basis, self.x[j])
                    scale *= (self.x[i] - self.x[j])
            c = self.y[i] / scale
            for k in range(len(basis)):
                coef[k] += c * basis[k]
        return coef

    def value(self, x_star: float) -> float:
        return poly_eval(self.coef, x_star)

    def polynomial_str(self) -> str:
        return "L(x) = " + poly_str(self.coef)

    def plot_data(self, step: float = 0.05) -> Tuple[List[float], List[float]]:
        x_min, x_max = min(self.x), max(self.x)
        count = int(round((x_max - x_min) / step))
        xs = [x_min + k * step for k in range(count + 1)]
        ys = [self.value(xv) for xv in xs]
        return xs, ys


# ---------- Ньютон ----------
class NewtonInterpolator:
    """Многочлен Ньютона через разделённые разности."""

    def __init__(self, x: List[float], y: List[float]):
        if len(x) != len(y):
            raise ValueError("Количество x и y должно совпадать")
        if len(x) < 2:
            raise ValueError("Нужно минимум 2 узла")
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.n = len(x)
        self.divided_diff = self._build_divided_diff_table()

    def _build_divided_diff_table(self) -> List[List[float]]:
        n = self.n
        table = [[0.0] * n for _ in range(n)]
        for i in range(n):
            table[i][0] = self.y[i]
        for j in range(1, n):
            for i in range(n - j):
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / \
                              (self.x[i + j] - self.x[i])
        return table

    def coefficients(self) -> List[float]:
        return [self.divided_diff[0][k] for k in range(self.n)]

    def divided_diff_table_str(self) -> str:
        header = f"{'x_i':>10} {'y[i]':>12}"
        for j in range(1, self.n):
            header += f" {'DD' + str(j):>14}"
        lines = [header]
        for i in range(self.n):
            line = f"{self.x[i]:>10.4f} {self.y[i]:>12.6f}"
            for j in range(1, self.n - i):
                line += f" {self.divided_diff[i][j]:>14.6f}"
            lines.append(line)
        return "\n".join(lines)

    def value(self, x_star: float) -> float:
        coefs = self.coefficients()
        n = len(coefs)
        res = coefs[n - 1]
        for k in range(n - 2, -1, -1):
            res = res * (x_star - self.x[k]) + coefs[k]
        return res

    @staticmethod
    def _factor(root: float) -> str:
        if abs(root) < 1e-12:
            return "(x)"
        if root > 0:
            return f"(x-{root:.6g})"
        return f"(x+{-root:.6g})"

    def polynomial_str(self) -> str:
        coefs = self.coefficients()
        terms = [f"{coefs[0]:.6g}"]
        prefix = ""
        for k in range(1, self.n):
            if k == 1:
                prefix = self._factor(self.x[0])
            else:
                prefix += self._factor(self.x[k - 1])
            terms.append(f"{coefs[k]:+.6g}*{prefix}")
        return "N(x) = " + " ".join(terms)

    def plot_data(self, step: float = 0.05) -> Tuple[List[float], List[float]]:
        x_min, x_max = min(self.x), max(self.x)
        count = int(round((x_max - x_min) / step))
        xs = [x_min + k * step for k in range(count + 1)]
        ys = [self.value(xv) for xv in xs]
        return xs, ys


# ---------- Сводка ----------
class InterpolationResults:
    def __init__(self, x, y, f, x_star):
        self.lagrange = LagrangeInterpolator(x, y)
        self.newton = NewtonInterpolator(x, y)
        self.lag_val = self.lagrange.value(x_star)
        self.new_val = self.newton.value(x_star)
        self.exact = f(x_star)
        self.abs_error = abs(self.lag_val - self.exact)
        self.diff_methods = abs(self.lag_val - self.new_val)

    def get_summary(self) -> dict:
        return {
            "lag_val": self.lag_val,
            "new_val": self.new_val,
            "exact": self.exact,
            "abs_error": self.abs_error,
            "diff": self.diff_methods,
        }
