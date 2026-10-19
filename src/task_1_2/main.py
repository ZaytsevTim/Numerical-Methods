import os
import sys

# Настраиваем пути, чтобы мини-мейн видел корень проекта src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from matrix_utils import TridiagonalReader
from tridiag_solver import TridiagonalSolver

def main():
    file_path = os.path.join(project_root, "..", "data", "tridiagonal.txt")
    
    print("=== 1.2 МЕТОД ПРОГОНКИ ДЛЯ ТРЁХДИАГОНАЛЬНОЙ СИСТЕМЫ ===")
    
    # Читаем данные через парсер
    parser = TridiagonalReader(file_path)
    N, main_diag, super_diag, sub_diag, d = parser.parse()
    
    print(f"Успешно прочитана трёхдиагональная система размера {N}x{N}\n")
    print(f"Главная диагональ (B): {main_diag}")
    print(f"Верхняя диагональ (C): {super_diag}")
    print(f"Нижняя диагональ  (A): {sub_diag}")
    print(f"Правая часть      (D): {d}\n")
    print("-" * 60)
    
    #Создаем экземпляр вычислительного класса
    solver = TridiagonalSolver(main_diag, super_diag, sub_diag, d)
    
    # Запускаем решение
    alpha, beta, x_solution = solver.solve()
    
    # Нада Вывести прогоночные коэффициенты
    print("Прогоночные коэффициенты:")
    for i in range(1, N):
        print(f"  Шаг {i}: alpha = {alpha[i]:.4f}, beta = {beta[i]:.4f}")
    print(f"  Финальный шаг: beta[N] = {beta[N]:.4f}\n")
    print("-" * 60)
    
    # И еще Вывести решение системы
    print(f"Решение системы x = {[round(elem, 4) for elem in x_solution]}")
    print("=" * 60)

if __name__ == "__main__":
    main()
