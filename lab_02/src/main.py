import os
import sys

# Настраиваем архитектурный мост путей до корня src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импортируем утилиты ввода, универсальные солверы и математику Варианта 10
from input_parsers import EquationParamReader, SystemParamReader
from nonlinear_functions import (
    nonlinear_system_domain_check,
    nonlinear_system_f,
    nonlinear_system_jacobian,
    nonlinear_system_phi,
    single_equation_domain_check,
    single_equation_f,
    single_equation_jacobian,
    single_equation_phi,
)
from system_solvers import UniversalMPISolver, UniversalNewtonSolver

def print_history_table(title, history):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)
    print(f"  {'Шаг':<5} | {'x1':<12} | {'x2':<12} | {'Невязка ||F||':<16} | {'Шаг ||dx||':<14}")
    print("  " + "-" * 74)
    for row in history:
        x1 = row['x'][0]
        x2 = row['x'][1] if len(row['x']) > 1 else float('nan')
        res = row['residual']
        sd = row['step_diff']
        sd_str = f"{sd:.3e}" if sd is not None else "—"
        print(f"  {row['iter']:<5} | {x1:<12.6f} | {x2:<12.6f} | {res:<16.3e} | {sd_str:<14}")

def main():
    # Задаем пути к конфигурационным текстовым файлам в изолированной папке data
    eq_file = os.path.abspath(os.path.join(project_root, "..", "lab_02", "data", "nonlinear_eq.txt"))
    sys_file = os.path.abspath(os.path.join(project_root, "..", "lab_02", "data", "nonlinear_sys.txt"))

    print("=" * 70)
    print("=== ЛАБОРАТОРНАЯ РАБОТА №2: РЕШЕНИЕ НЕЛИНЕЙНЫХ УРАВНЕНИЙ И СИСТЕМ ===")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # ВЫПОЛНЕНИЕ ПУНКТА 2.1 (Одиночное нелинейное уравнение)
    # --------------------------------------------------------------------------
    print("\n>>> Выполнение пункта 2.1 (Одиночное тригонометрическое уравнение) <<<")
    eps_21, x0_21 = EquationParamReader(eq_file).parse()
    print(f"  Входные данные загружены: x0 = {x0_21}, точность eps = {eps_21}")

    # Запуск универсального метода Ньютона на размере 1x1
    newton_solver_21 = UniversalNewtonSolver(
        single_equation_f,
        single_equation_jacobian,
        eps_21,
        domain_check=single_equation_domain_check
    )
    root_newton_21, hist_newton_21 = newton_solver_21.solve(x0_21)
    print(f"  [Метод Ньютона] Найдено решение: x = {root_newton_21[0]:.6f} за {len(hist_newton_21)} итераций")

    # Запуск универсального метода простых итераций (МПИ)
    mpi_solver_21 = UniversalMPISolver(
        single_equation_f,
        single_equation_phi,
        eps_21,
        domain_check=single_equation_domain_check
    )
    root_mpi_21, hist_mpi_21 = mpi_solver_21.solve(x0_21)
    print(f"  [Метод МПИ]     Найдено решение: x = {root_mpi_21[0]:.6f} за {len(hist_mpi_21)} итераций")

    # --------------------------------------------------------------------------
    # ВЫПОЛНЕНИЕ ПУНКТА 2.2 (Система нелинейных уравнений)
    # --------------------------------------------------------------------------
    print("\n\n>>> Выполнение пункта 2.2 (Нелинейная тригонометрическая система) <<<")
    n_22, eps_22, x0_22 = SystemParamReader(sys_file).parse()
    print(f"  Входные данные загружены: размерность N = {n_22}, X0 = {x0_22}, точность eps = {eps_22}")

    # Запуск системного метода Ньютона с интеграцией нашего LUP-разложения
    newton_solver_22 = UniversalNewtonSolver(
        nonlinear_system_f,
        nonlinear_system_jacobian,
        eps_22,
        domain_check=nonlinear_system_domain_check
    )
    root_newton_22, hist_newton_22 = newton_solver_22.solve(x0_22)
    print(f"  [Метод Ньютона] Решение системы: x1 = {root_newton_22[0]:.6f}, x2 = {root_newton_22[1]:.6f}")
    print(f"                  Сходимость достигнута за {len(hist_newton_22)} шагов")

    # Запуск системного МПИ
    mpi_solver_22 = UniversalMPISolver(
        nonlinear_system_f,
        nonlinear_system_phi,
        eps_22,
        domain_check=nonlinear_system_domain_check
    )
    root_mpi_22, hist_mpi_22 = mpi_solver_22.solve(x0_22)
    print(f"  [Метод МПИ]     Решение системы: x1 = {root_mpi_22[0]:.6f}, x2 = {root_mpi_22[1]:.6f}")
    print(f"                  Сходимость достигнута за {len(hist_mpi_22)} шагов")

    # --------------------------------------------------------------------------
    # АНАЛИЗ ЗАВИСИМОСТИ ПОГРЕШНОСТИ ОТ КОЛИЧЕСТВА ИТЕРАЦИЙ (ДЛЯ ОТЧЕТА ПО ТЗ)
    # --------------------------------------------------------------------------

    print_history_table("МЕТОД НЬЮТОНА (пункт 2.2): траектория и сходимость", hist_newton_22)
    print_history_table("МЕТОД ПРОСТЫХ ИТЕРАЦИЙ (пункт 2.2): траектория и сходимость", hist_mpi_22)

    print("=" * 70)

if __name__ == "__main__":
    main()
