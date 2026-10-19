import os
import sys

# Строим архитектурный мост путей до корня src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from matrix_utils import SystemEpsReader, matrix_multiply, is_matrix_equal
from qr_solver import QRDecompositionSolver


def print_matrix(name, matrix):
    """ Утилита для красивого форматированного вывода матриц """
    print(f"{name} =")
    for row in matrix:
        print("  " + " ".join(f"{elem:8.4f}" for elem in row))


def main():
    file_path = os.path.abspath(os.path.join(project_root, "..", "data", "qr.txt"))

    print("=" * 60)
    print("=== 1.5 QR-РАЗЛОЖЕНИЕ МЕТОДОМ ОТРАЖЕНИЙ ХАУСХОЛДЕРА ===")
    print("=" * 60)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл данных не найден по пути: {file_path}")

    # Шаг 1: Читаем исходную матрицу через ООП-парсер
    parser = SystemEpsReader(file_path)
    N, A, _, eps = parser.parse()

    print(f"Успешно прочитана матрица размера {N}x{N}. Точность eps = {eps}\n")
    print_matrix("Исходная матрица A", A)
    print("-" * 60)

    # Шаг 2: Создаем объект вычислительного класса
    solver = QRDecompositionSolver(A, eps)

    # Требование ТЗ: демонстрация однократного разложения на Q и R
    # Берём самое первое разложение исходной матрицы A
    Q_first, R_first = solver.householder_qr(A)

    print("ВЕРИФИКАЦИЯ ОДНОКРАТНОГО QR-РАЗЛОЖЕНИЯ (НА СТАРТЕ):")
    print_matrix("Ортогональная матрица Q (зеркала Хаусхолдера)", Q_first)
    print_matrix("Верхняя треугольная матрица R", R_first)
    print("-" * 60)

    # Требование ТЗ: проверка результата обратного умножения Q * R
    qr_prod = matrix_multiply(Q_first, R_first)
    print_matrix("Результат обратного перемножения Q * R", qr_prod)
    print(f"Проверка корректности разложения (Q * R == A): "
          f"{is_matrix_equal(qr_prod, A, eps=1e-5)}")
    print("-" * 60)

    # Шаг 3: Запускаем полный итерационный цикл для поиска собственных чисел
    eigenvalues, total_iters = solver.solve()

    print(f"QR-алгоритм завершён за {total_iters} глобальных итераций.\n")

    # Требование ТЗ: вывод найденных собственных значений
    print("Найденные собственные значения (лямбды):")
    for i, val in enumerate(eigenvalues):
        if abs(val.imag) < 1e-9:
            print(f"  lambda_{i + 1} = {val.real:.4f}")
        else:
            sign = "+" if val.imag >= 0 else "-"
            print(f"  lambda_{i + 1} = {val.real:.4f} {sign} {abs(val.imag):.4f}i")
    print("=" * 60)


if __name__ == "__main__":
    main()