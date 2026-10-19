import os
import sys

# Настраиваем пути (наш мост): поднимаем область видимости Питона до корня 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Теперь Питон идеально видит и утилиты из корня, и решатель из текущей папки
from matrix_utils import SystemReader, matrix_multiply, is_matrix_equal
from lu_decomposition import LUPDecomposition

def print_matrix(name, matrix):
    """ Утилита для красивого вывода матриц в консоль """
    print(f"{name} =")
    for row in matrix:
        print("  " + " ".join(f"{elem:8.4f}" for elem in row))

def main():
    # Шагаем из папки task_1_1 наружу в корень, а затем в общую папку data
    file_path = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "LU.txt"))

    print("=" * 60)
    print("=== 1.1 LUP-РАЗЛОЖЕНИЕ И РЕШЕНИЕ СИСТЕМЫ (МЕТОД ГАУССА) ===")
    print("=" * 60)
    
    # Шаг 1: Полиморфное чтение данных через наш базовый парсер
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Не найден файл с данными по пути: {file_path}")
        
    parser = SystemReader(file_path)
    N, A, b = parser.parse()
    
    print(f"Успешно прочитана система уравнений размера {N}x{N}\n")
    print_matrix("Исходная матрица A", A)
    print(f"Вектор свободных членов b = {b}\n")
    print("-" * 60)
    
    # Шаг 2: Создаем объект разложения (все вычисления L, U, P идут внутри __init__)
    lu_solver = LUPDecomposition(A)
    
    # Требование 1 ТЗ: Вывод полученных матриц L, U, P
    print_matrix("Нижняя треугольная матрица L (История Гаусса)", lu_solver.L)
    print_matrix("Верхняя треугольная матрица U (Ступенчатый вид)", lu_solver.U)
    print_matrix("Матрица перестановок P (Блокнот пивотинга)", lu_solver.P)
    print(f"Количество сделанных перестановок строк: {lu_solver.swaps_count}")
    print("-" * 60)
    
    # Требование 2 ТЗ: Результат произведения L * U
    lu_prod = matrix_multiply(lu_solver.L, lu_solver.U)
    print_matrix("Результат обратного перемножения L * U", lu_prod)
    print("-" * 60)
    
    # Требование 3 ТЗ: Решение системы Ax = b через треугольные подстановки
    x_solution = lu_solver.solve(b)
    print(f"Решение системы уравнений x = {[round(elem, 4) for elem in x_solution]}")
    print("-" * 60)
    
    # Требование 4 ТЗ: Нахождение обратной матрицы A^-1 по столбцам
    A_inv = lu_solver.get_inverse()
    print_matrix("Обратная матрица A^-1", A_inv)
    print("-" * 60)
    
    # Требование 5 ТЗ: Нахождение определителя матрицы det(A) с учетом знака
    det_A = lu_solver.get_determinant()
    print(f"Определитель матрицы det(A) = {det_A:.4f}")
    print("-" * 60)
    
    # Требование 6 ТЗ: Проверка свойства обратной матрицы A * A^-1 = E
    a_ainv_prod = matrix_multiply(A, A_inv)
    print_matrix("Проверка: произведение исходной матрицы на обратную A * A^-1", a_ainv_prod)
    
    # Генерируем эталонную единичную матрицу E для автоматической проверки
    E = [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]
    print(f"A * A^-1 == E: {is_matrix_equal(a_ainv_prod, E, eps=1e-5)}")
    print("-" * 60)
    
    # Требование 7 ТЗ: Проверка корректности разложения L * U = P * A
    pa_prod = matrix_multiply(lu_solver.P, A)
    print_matrix("Произведение P * A", pa_prod)
    print(f"Результат проверки (L * U == P * A): {is_matrix_equal(lu_prod, pa_prod)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
