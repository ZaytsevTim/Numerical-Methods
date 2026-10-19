import os
import sys

# Наш архитектурный мост путей до корня 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импортируем утилиты, текущий итерационный класс, а также ТОЧНЫЙ LUP из папки 1.1!
from matrix_utils import SystemEpsReader
from iter_solver import IterationSolver
from task_1_1.lu_decomposition import LUPDecomposition

def main():
    # Шагаем в общую папку data за файлом для итераций
    file_path = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "iterations.txt"))

    print("=" * 60)
    print("=== 1.3 МЕТОД ПРОСТЫХ ИТЕРАЦИЙ И МЕТОД ЗЕЙДЕЛЯ ===")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Не найден файл с данными по пути: {file_path}")
        
    # 1. Читаем данные через наш ООП-парсер (он вытащит и матрицу, и вектор, и эпсилон)
    parser = SystemEpsReader(file_path)
    N, A, b, eps = parser.parse()
    
    print(f"Успешно прочитана система размера {N}x{N} с точностью eps = {eps}\n")
    print(f"Вектор b = {b}\n")
    print("-" * 60)
    
    # 2. Создаем объект нашего DRY-решателя
    solver = IterationSolver(A, b, eps)
    
    # 3. Запускаем оба метода и получаем ответы + число шагов
    x_jacobi, iter_jacobi = solver.solve_jacobi()
    x_seidel, iter_seidel = solver.solve_seidel()
    
    # 4. Вычисляем ЭТАЛОННОЕ решение через наш готовый LUP-класс из первой лабы!
    lu_exact_solver = LUPDecomposition(A)
    x_exact = lu_exact_solver.solve(b)
    
    # Требование ТЗ: Вывести результаты и сравнить количество итераций
    print("Результаты вычислений:")
    print(f"  МПИ (Якоби):   x = {[round(elem, 4) for elem in x_jacobi]} (сошёлся за {iter_jacobi} шагов)")
    print(f"  Метод Зейделя: x = {[round(elem, 4) for elem in x_seidel]} (сошёлся за {iter_seidel} шагов)")
    print("-" * 60)
    
    # Требование ТЗ: Вывести эталонное решение для проверки результатов
    print(f"  ЭТАЛОН (LUP):  x = {[round(elem, 4) for elem in x_exact]} (Точный аналитический метод)")
    print("-" * 60)
    
    # Сравниваем Зейделя и Якоби наглядно
    if iter_seidel < iter_jacobi:
        print(f"Вывод по ТЗ: Метод Зейделя сошёлся БЫСТРЕЕ МПИ на {iter_jacobi - iter_seidel} итераций за счет немедленного вкрапления свежих компонент!")
    else:
        print("Вывод по ТЗ: Скорость методов оказалась равной на данной матрице.")
    print("=" * 60)

if __name__ == "__main__":
    main()
