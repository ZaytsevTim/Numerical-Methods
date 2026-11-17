import math
import os
import sys

# Настраиваем архитектурный мост путей
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_reader import (
    read_differentiation_data,
    read_interpolation_data,
    read_least_squares_data,
    read_spline_data,
    read_integration_data,
)
from differentiation import NumericalDifferentiation
from graphs import GraphBuilder
from interpolation import InterpolationResults
from integration import IntegrationResults
from least_squares import create_least_squares_results
from spline import CubicSpline


# =============================================================================
# ПУНКТ 3.1: ИНТЕРПОЛЯЦИЯ ЛАГРАНЖА И НЬЮТОНА
# =============================================================================
def task_3_1(data_dir):
    print("\n" + "=" * 70)
    print("=== 3.1 ИНТЕРПОЛЯЦИОННЫЕ МНОГОЧЛЕНЫ ЛАГРАНЖА И НЬЮТОНА ===")
    print("=" * 70)

    data = read_interpolation_data(data_dir)
    x, y, x_star = data.parse()

    # ВАРИАНТ 10: Функция y = arctg(x)
    def f(x_val):
        return math.atan(x_val)

    results = InterpolationResults(x, y, f, x_star)
    ir = results.get_summary()

    print(f"\nИсходные узлы: x = {[round(v, 4) for v in x]}")
    print(f"Значения:     y = {[round(v, 4) for v in y]}")
    print(f"Точка x* = {x_star:.4f}")
    print("-" * 70)

    print("\nМногочлен Лагранжа (в явном виде):")
    print(results.lagrange.polynomial_str())

    print("\nТаблица разделённых разностей:")
    print(results.newton.divided_diff_table_str())

    print("\nМногочлен Ньютона:")
    print(results.newton.polynomial_str())
    print("-" * 70)

    print(f"\nЗначение Лагранжа в x*: {ir['lag_val']:.10f}")
    print(f"Значение Ньютона в x*:  {ir['new_val']:.10f}")
    print(f"Точное значение f(x*):  {ir['exact']:.10f}")
    print(f"Абсолютная погрешность: {ir['abs_error']:.2e}")
    print(f"Разность методов:       {ir['diff']:.2e}")

    # Построение графика
    lag_xs, lag_ys = results.lagrange.plot_data()
    new_xs, new_ys = results.newton.plot_data()

    f_xs, f_ys = [], []
    x_min, x_max = min(x), max(x)
    step = 0.05
    count = int(round((x_max - x_min) / step))
    for k in range(count + 1):
        xv = x_min + k * step
        f_xs.append(xv)
        f_ys.append(f(xv))

    builder = GraphBuilder("3.1: Интерполяция Лагранжа и Ньютона (Вариант 10)")
    builder.build_interpolation_graph(
        x, y, x_star,
        lag_xs, lag_ys, new_xs, new_ys,
        f_ys, f_xs, ir['lag_val']
    )
    builder.save(os.path.join(data_dir, "lab3_31.png"))


# =============================================================================
# ПУНКТ 3.2: КУБИЧЕСКИЙ СПЛАЙН
# =============================================================================
def task_3_2(data_dir):
    print("\n" + "=" * 70)
    print("=== 3.2 КУБИЧЕСКИЙ СПЛАЙН ===")
    print("=" * 70)

    data = read_spline_data(data_dir)
    x, y, x_star = data.parse()

    spline = CubicSpline(x, y)
    spline_value, interval_idx = spline.value(x_star)

    print(f"\nИсходные узлы: x = {[round(v, 4) for v in x]}")
    print(f"Значения:     y = {[round(v, 4) for v in y]}")
    print(f"Точка x* = {x_star:.4f}")
    print("-" * 70)

    print("\nКоэффициенты сплайна:")
    print(spline.coefficients_str())
    print("-" * 70)

    print(f"\nЗначение сплайна в x*: S(x*) = {spline_value:.6f}")
    print(f"Номер интервала: {interval_idx} "
          f"[{x[interval_idx]:.4f}, {x[interval_idx + 1]:.4f}]")
    print("-" * 70)

    check = spline.check_nodes()
    print(f"\nПроверка прохождения через узлы (макс.): {max(check):.2e}")

    c0, cn = spline.boundary_second_derivatives()
    print(f"Вторая производная на концах: "
          f"S''(x0) = {c0:.2e}, S''(xn) = {cn:.2e}")
    print("(для естественного сплайна ~0)")

    spline_xs, spline_ys = spline.plot_data()
    builder = GraphBuilder("3.2: Кубический сплайн (Вариант 10)")
    builder.build_spline_graph(x, y, spline_xs, spline_ys, x_star, spline_value)
    builder.save(os.path.join(data_dir, "lab3_32.png"))


# =============================================================================
# ПУНКТ 3.3: МЕТОД НАИМЕНЬШИХ КВАДРАТОВ
# =============================================================================
def task_3_3(data_dir):
    print("\n" + "=" * 70)
    print("=== 3.3 МЕТОД НАИМЕНЬШИХ КВАДРАТОВ ===")
    print("=" * 70)

    data = read_least_squares_data(data_dir)
    x, y = data.parse()

    results = create_least_squares_results(x, y)
    p1, p2, p3 = results["p1"], results["p2"], results["p3"]
    mse1, mse2, mse3 = results["mse1"], results["mse2"], results["mse3"]

    print(f"\nЧисло точек: n = {len(x)}")
    print(f"Коэффициенты P1: {[round(c, 4) for c in p1.coeffs]}")
    print(f"Коэффициенты P2: {[round(c, 4) for c in p2.coeffs]}")
    print(f"Коэффициенты P3: {[round(c, 4) for c in p3.coeffs]}")
    print("-" * 70)

    print(f"\nP1(x) = {p1.polynomial_str()}")
    print(f"P2(x) = {p2.polynomial_str()}")
    print(f"P3(x) = {p3.polynomial_str()}")
    print("-" * 70)

    print(f"\nMSE1 = {mse1:.6f}")
    print(f"MSE2 = {mse2:.6f}")
    print(f"MSE3 = {mse3:.6f}")

    errors = {"P1": mse1, "P2": mse2, "P3": mse3}
    best_p = min(errors, key=lambda k: errors[k])
    print(f"\nЛучшее приближение: {best_p} (MSE = {errors[best_p]:.6f})")

    p1_xs, p1_ys = p1.plot_data()
    p2_xs, p2_ys = p2.plot_data()
    p3_xs, p3_ys = p3.plot_data()
    builder = GraphBuilder("3.3: Метод наименьших квадратов (Вариант 10)")
    builder.build_least_squares_graph(x, y, p1_xs, p1_ys, p2_xs, p2_ys, p3_xs, p3_ys)
    builder.save(os.path.join(data_dir, "lab3_33.png"))


# =============================================================================
# ПУНКТ 3.4: ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ
# =============================================================================
def task_3_4(data_dir):
    print("\n" + "=" * 70)
    print("=== 3.4 ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ ===")
    print("=" * 70)

    data = read_differentiation_data(data_dir)
    x, y, x_star = data.parse()

    diff = NumericalDifferentiation(x, y)

    print("\nИсходная таблица значений:")
    for xi, yi in zip(x, y):
        print(f"  x = {xi:.4f}, f(x) = {yi:.4f}")
    print(f"Точка вычисления: x* = {x_star:.4f}")
    print(f"Использованные узлы: {diff.nodes_used_str(x_star)}")
    print("-" * 70)

    f1 = diff.first_derivative(x_star)
    f2 = diff.second_derivative(x_star)
    print(f"\nПервая производная  f'(x*)  = {f1:.6f}")
    print(f"Вторая производная  f''(x*) = {f2:.6f}")


# =============================================================================
# ПУНКТ 3.5: ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ
# =============================================================================
def task_3_5(data_dir):
    print("\n" + "=" * 70)
    print("=== 3.5 ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ ===")
    print("=" * 70)

    data = read_integration_data(data_dir)
    a, b, h1, h2, _ = data.parse()

    # ВАРИАНТ 10: y = x^2 / (x^2 + 16)
    def func(x_val):
        return (x_val ** 2) / (x_val ** 2 + 16.0)

    # Аналитический первообразный интеграл: x - 4*arctg(x/4)
    def exact_func(x_val):
        return x_val - 4.0 * math.atan(x_val / 4.0)

    exact = exact_func(b) - exact_func(a)

    results = IntegrationResults(func, a, b, exact=exact, h1=h1, h2=h2)

    print(f"\nИнтеграл функции x^2/(x^2+16) от {a} до {b}")
    print(f"Шаги: h1 = {h1}, h2 = {h2}")
    print("-" * 70)

    print(results.get_table_str())
    print("-" * 70)

    if exact is not None:
        print(f"\nТочное значение интеграла: {exact:.8f}")
        best = results.best_method()
        print(f"Наиболее точный метод: {best['name']} (абс. ошибка {best['abs_error']:.2e})")


# =============================================================================
# MAIN
# =============================================================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lab03_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(lab03_root, "data")

    print("=" * 70)
    print("=== ЛАБОРАТОРНАЯ РАБОТА №3 (ВАРИАНТ 10) ===")
    print("=" * 70)
    print(f"Каталог с данными: {data_dir}\n")

    task_3_1(data_dir)
    task_3_2(data_dir)
    task_3_3(data_dir)
    task_3_4(data_dir)
    task_3_5(data_dir)

    print("\n" + "=" * 70)
    print("Все задачи выполнены!")
    print("=" * 70)


if __name__ == "__main__":
    main()
