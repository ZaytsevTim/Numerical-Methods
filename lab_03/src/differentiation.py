from typing import List, Tuple


class NumericalDifferentiation:
    """Численное дифференцирование по таблице значений функции.

    Строит интерполяционный многочлен Ньютона по ближайшим к x* узлам
    и дифференцирует его аналитически. Работает и для неравномерной сетки.

    Для первой и второй производной используется один и тот же набор
    из 3 узлов, поэтому в выводе «использованные узлы» точно соответствуют
    тем, что реально участвуют в расчёте (требование ТЗ).
    """

    def __init__(self, x: List[float], y: List[float]):
        if len(x) != len(y):
            raise ValueError("Количество узлов x и y должно совпадать")
        if len(x) < 3:
            raise ValueError("Нужно минимум 3 узла для дифференцирования")
        self.x = [float(xi) for xi in x]
        self.y = [float(yi) for yi in y]
        self.n = len(x)

    # ------------------------------------------------------------------
    # Выбор узлов
    # ------------------------------------------------------------------
    def _select_nodes(self, x_star: float, count: int = 3
                      ) -> Tuple[List[float], List[float]]:
        """Выбирает `count` узлов так, чтобы x* был как можно ближе
        к центру набора (улучшает точность производных).

        Для x* внутри таблицы рассматриваются два варианта положения
        набора (со сдвигом влево/вправо) и выбирается тот, где центральный
        узел ближе к x*.
        """
        # Индекс интервала, содержащего x*
        idx = 0
        for i in range(self.n - 1):
            if self.x[i] <= x_star <= self.x[i + 1]:
                idx = i
                break
        else:
            idx = self.n - 2 if x_star > self.x[-1] else 0

        half = count // 2

        # Два кандидата на начало набора: x* чуть левее или чуть правее центра
        best_start = None
        best_dist = None
        for start in (idx - half, idx - half + 1):
            start = max(0, min(start, self.n - count))
            center = self.x[start + half]
            dist = abs(center - x_star)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_start = start

        return (self.x[best_start:best_start + count],
                self.y[best_start:best_start + count])

    # ------------------------------------------------------------------
    # Разделённые разности
    # ------------------------------------------------------------------
    @staticmethod
    def _divided_differences(xn: List[float], yn: List[float]
                             ) -> List[float]:
        """Коэффициенты Ньютона: f[x0], f[x0,x1], ..."""
        m = len(xn)
        table = [[0.0] * m for _ in range(m)]
        for i in range(m):
            table[i][0] = yn[i]
        for j in range(1, m):
            for i in range(m - j):
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / \
                              (xn[i + j] - xn[i])
        return [table[0][k] for k in range(m)]

    # ------------------------------------------------------------------
    # Раскрытие формы Ньютона в мономиальный базис
    # ------------------------------------------------------------------
    @staticmethod
    def _newton_to_monomial(coefs: List[float], xn: List[float]) -> List[float]:
        """Преобразует N(x) = sum c_k * prod_{j<k}(x - x_j) в сумму p_i x^i.

        Возвращает коэффициенты p_0..p_{m-1} (от младшей степени к старшей).
        """
        m = len(coefs)
        poly = [0.0] * m
        poly[0] = coefs[-1]
        for k in range(m - 2, -1, -1):
            new_poly = [0.0] * m
            for i in range(m):
                if poly[i] == 0.0:
                    continue
                new_poly[i] += -xn[k] * poly[i]
                if i + 1 < m:
                    new_poly[i + 1] += poly[i]
            new_poly[0] += coefs[k]
            poly = new_poly
        return poly

    # ------------------------------------------------------------------
    # Значение и производные (схема Горнера)
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_with_derivs(poly: List[float], x_star: float
                          ) -> Tuple[float, float, float]:
        """Возвращает (P(x*), P'(x*), P''(x*)) по схеме Горнера.

        poly — коэффициенты от младшей степени к старшей.
        """
        m = len(poly)

        # b: для P(x)
        b = [0.0] * m
        b[m - 1] = poly[m - 1]
        for k in range(m - 2, -1, -1):
            b[k] = poly[k] + x_star * b[k + 1]
        value = b[0]

        # c: для P'(x)
        if m >= 2:
            c = [0.0] * (m - 1)
            c[m - 2] = b[m - 1]
            for k in range(m - 3, -1, -1):
                c[k] = b[k + 1] + x_star * c[k + 1]
            d1 = c[0]
        else:
            d1 = 0.0

        # d: для P''(x)
        if m >= 3:
            d = [0.0] * (m - 2)
            d[m - 3] = 2.0 * c[m - 2]
            for k in range(m - 4, -1, -1):
                d[k] = 2.0 * c[k + 1] + x_star * d[k + 1]
            d2 = d[0]
        else:
            d2 = 0.0

        return value, d1, d2

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------
    def _derivatives(self, x_star: float, count: int
                     ) -> Tuple[float, float]:
        xn, yn = self._select_nodes(x_star, count=count)
        coefs = self._divided_differences(xn, yn)
        poly = self._newton_to_monomial(coefs, xn)
        _, d1, d2 = self._eval_with_derivs(poly, x_star)
        return d1, d2

    def first_derivative(self, x_star: float) -> float:
        d1, _ = self._derivatives(x_star, count=3)
        return d1

    def second_derivative(self, x_star: float) -> float:
        # 3 узла — квадратичный Ньютон, N'' постоянна на наборе.
        # Это эквивалентно классической центральной разности
        #     f''(x_i) ≈ (f_{i+1} - 2f_i + f_{i-1}) / h²
        # и гарантирует, что «использованные узлы» в выводе (3 шт.)
        # действительно совпадают с теми, что участвуют в расчёте f''.
        _, d2 = self._derivatives(x_star, count=3)
        return d2

    def nodes_used_str(self, x_star: float) -> str:
        xn, yn = self._select_nodes(x_star, count=3)
        return ", ".join(f"({xi:.4f}, {yi:.4f})" for xi, yi in zip(xn, yn))
