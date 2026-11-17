"""3.5: Численное интегрирование."""
import math
from typing import Optional, List, Dict


class IntegrationMethod:
    """Базовый класс метода интегрирования."""
    name = "base"
    order_p = 1

    def __init__(self, func, a: float, b: float, n: int):
        self.func = func
        self.a = a
        self.b = b
        self.n = n
        self.h = (b - a) / n

    def compute(self) -> float:
        raise NotImplementedError


class LeftRectangles(IntegrationMethod):
    name = "Левые прямоугольники"
    order_p = 1

    def compute(self) -> float:
        s = sum(self.func(self.a + i * self.h) for i in range(self.n))
        return s * self.h


class RightRectangles(IntegrationMethod):
    name = "Правые прямоугольники"
    order_p = 1

    def compute(self) -> float:
        s = sum(self.func(self.a + (i + 1) * self.h) for i in range(self.n))
        return s * self.h


class MidpointRectangles(IntegrationMethod):
    name = "Средние прямоугольники"
    order_p = 2

    def compute(self) -> float:
        s = sum(self.func(self.a + (i + 0.5) * self.h) for i in range(self.n))
        return s * self.h


class Trapezoid(IntegrationMethod):
    name = "Трапеции"
    order_p = 2

    def compute(self) -> float:
        s = 0.5 * (self.func(self.a) + self.func(self.b))
        s += sum(self.func(self.a + i * self.h) for i in range(1, self.n))
        return s * self.h


class Simpson(IntegrationMethod):
    name = "Симпсон"
    order_p = 4

    def __init__(self, func, a: float, b: float, n: int):
        # Гарантируем чётное n — иначе формула Симпсона неприменима
        if n % 2 != 0:
            n += 1
        super().__init__(func, a, b, n)

    def compute(self) -> float:
        s = self.func(self.a) + self.func(self.b)
        for i in range(1, self.n):
            s += (4.0 if i % 2 == 1 else 2.0) * self.func(self.a + i * self.h)
        return s * self.h / 3.0


class IntegrationResults:
    """Считает интеграл всеми методами и применяет Рунге–Ромберга."""

    METHOD_CLASSES = [
        LeftRectangles, RightRectangles, MidpointRectangles, Trapezoid, Simpson
    ]

    def __init__(self, func, a: float, b: float, exact=None,
                 h1: float = None, h2: float = None):
        self.func = func
        self.a = a
        self.b = b
        self.exact = exact

        # n из шагов (минимум 1)
        self.n1 = max(1, int(round((b - a) / h1))) if h1 else 1
        self.n2 = max(1, int(round((b - a) / h2))) if h2 else 2

        # Уточняющий шаг должен быть меньше: n2 > n1
        if self.n2 <= self.n1:
            self.n2 = self.n1 * 2

        self.results = []

        for cls in self.METHOD_CLASSES:
            m1 = cls(func, a, b, self.n1)
            m2 = cls(func, a, b, self.n2)

            res_h1 = m1.compute()
            res_h2 = m2.compute()
            p = cls.order_p

            # Рунге–Ромберг: I_RR = I(h2) + (I(h2) - I(h1)) / (2^p - 1)
            runge_romberg = res_h2 + (res_h2 - res_h1) / (2 ** p - 1)

            # Оценка погрешности: |I(h2) - I(h1)| / (2^p - 1)
            error_h2 = abs(res_h2 - res_h1) / (2 ** p - 1)

            abs_err = abs(res_h2 - exact) if exact is not None else None

            self.results.append({
                "name": cls.name,
                "n1": m1.n,          # фактическое n (может быть скорректировано)
                "n2": m2.n,
                "h1_val": res_h1,
                "h2_val": res_h2,
                "p": p,
                "runge_romberg": runge_romberg,
                "error_h2": error_h2,
                "abs_error": abs_err,
            })

    def get_table_str(self) -> str:
        lines = []
        lines.append(
            f"{'Метод':<22} | {'I(h1)':>12} | {'I(h2)':>12} | {'p':>2} | "
            f"{'Рунге-Ромб.':>12} | {'Ошибка(h2)':>10} | {'Абс.погр.':>10}"
        )
        lines.append("-" * 106)
        for r in self.results:
            abs_err = f"{r['abs_error']:.2e}" if r['abs_error'] is not None else "—"
            lines.append(
                f"{r['name']:<22} | {r['h1_val']:>12.6f} | {r['h2_val']:>12.6f} | "
                f"{r['p']:>2} | {r['runge_romberg']:>12.6f} | "
                f"{r['error_h2']:>10.2e} | {abs_err:>10}"
            )
        return "\n".join(lines)

    def best_method(self) -> dict:
        """Метод с минимальной погрешностью (абсолютной, если есть exact)."""
        if self.exact is None:
            return min(self.results, key=lambda r: r["error_h2"])
        return min(self.results, key=lambda r: r["abs_error"])
