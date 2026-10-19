import os
import sys

# Строим мост путей до корня src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from matrix_utils import SystemEpsReader, matrix_multiply, is_matrix_equal
from rotate_solver import JacobiRotationSolver

def print_matrix(name, matrix):
    """ Красивый вывод матриц в консоль """
    print(f"{name} =")
    for row in matrix:
        print("  " + " ".join(f"{elem:8.4f}" for elem in row))

def main():
    # Идем в папку data за тестом
    file_path = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "jacobi_rotate.txt"))

    print("=" * 60)
    print("=== 1.4 МЕТОД ВРАЩЕНИЙ ЯКОБИ С ПРОВЕРКОЙ AV == V*LAMBDA ===")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Не найден файл с данными по пути: {file_path}")
        
    # Читаем данные через наш ООП-парсер
    parser = SystemEpsReader(file_path)
    N, A, _, eps = parser.parse()
    
    print(f"Успешно прочитана симметричная матрица размера {N}x{N} с точностью eps = {eps}\n")
    print_matrix("Исходная матрица A", A)
    print("-" * 60)
    
    # Запускаем карусель вращений Якоби
    solver = JacobiRotationSolver(A, eps)
    lambdas, V_matrix, steps = solver.solve()
    
    print(f"Карусель остановилась через {steps} поворотов.\n")
    print("Найденные собственные значения (лямбды):")
    for i, lam in enumerate(lambdas):
        print(f"  lambda_{i+1} = {lam:.4f}")
    print("-" * 60)
    
    print_matrix("Матрица собственных векторов V", V_matrix)
    print("-" * 60)
    
    # === ТРЕБОВАНИЕ ТЗ: ПРОВЕРКА КОРРЕКТНОСТИ AV == V * LAMBDA ===
    
    # 1. Считаем левую часть: Произведение A * V
    av_prod = matrix_multiply(A, V_matrix)
    print_matrix("Левая часть: Произведение A * V", av_prod)
    print("-" * 60)
    
    # 2. Составляем диагональную матрицу собственных значений Lambda
    Lambda_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        Lambda_matrix[i][i] = lambdas[i]
        
    # 3. Считаем правую часть: Произведение V * Lambda
    vlam_prod = matrix_multiply(V_matrix, Lambda_matrix)
    print_matrix("Правая часть: Произведение V * Lambda", vlam_prod)
    print("-" * 60)
    
    # 4. Автоматическая верификация равенства матриц с учетом погрешности eps
    is_correct = is_matrix_equal(av_prod, vlam_prod, eps=1e-5)
    print(f"Результат проверки равенства (A * V == V * Lambda): {is_correct}")
    print("=" * 60)

if __name__ == "__main__":
    main()
