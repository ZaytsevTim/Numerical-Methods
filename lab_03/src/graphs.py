"""Построение графиков лабораторной работы №3."""
from typing import List, Tuple
import matplotlib.pyplot as plt


class GraphBuilder:
    """Помощник для построения графиков."""

    def __init__(self, title: str, figsize: Tuple[int, int] = (10, 6)):
        self.title = title
        self.figsize = figsize
        self.fig, self.ax = None, None

    def build_interpolation_graph(
        self, x, y, x_star, lagrange_xs, lagrange_ys,
        newton_xs, newton_ys, f_ys, f_xs, lag_val
    ):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.plot(f_xs, f_ys, label='Исходная функция f(x)', color='blue')
        self.ax.plot(lagrange_xs, lagrange_ys, label='Многочлен Лагранжа',
                     linestyle='--', color='green')
        self.ax.plot(newton_xs, newton_ys, label='Многочлен Ньютона',
                     linestyle=':', color='orange')
        self.ax.scatter(x, y, color='red', zorder=5, label='Узлы интерполяции')
        self.ax.scatter([x_star], [lag_val], color='purple',
                        marker='*', s=200, zorder=6, label='Заданная точка x*')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(self.title)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        return self.fig

    def build_spline_graph(self, x, y, spline_xs, spline_ys, x_star, spline_value):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.plot(spline_xs, spline_ys, label='Кубический сплайн',
                     color='blue', linewidth=2)
        self.ax.scatter(x, y, color='red', zorder=5, label='Исходные узлы')
        self.ax.scatter([x_star], [spline_value], color='purple',
                        marker='*', s=200, zorder=6, label='Заданная точка (x*, S(x*))')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(self.title)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        return self.fig

    def build_least_squares_graph(self, x, y, p1_xs, p1_ys, p2_xs, p2_ys, p3_xs, p3_ys):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.scatter(x, y, color='black', zorder=5, label='Исходные данные')
        self.ax.plot(p1_xs, p1_ys, label='P1(x) - степень 1', linestyle='--', color='blue')
        self.ax.plot(p2_xs, p2_ys, label='P2(x) - степень 2', linestyle='--', color='green')
        self.ax.plot(p3_xs, p3_ys, label='P3(x) - степень 3', linestyle='--', color='red')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(self.title)
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        return self.fig

    def save(self, filename):
        if self.fig is None:
            print(f"ВНИМАНИЕ: График не был построен, сохранять нечего!")
            return
        self.fig.savefig(filename)
        plt.close(self.fig)
        print(f"  [График сохранен: {filename}]")
