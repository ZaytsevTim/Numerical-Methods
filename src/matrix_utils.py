class BaseMatrixReader:
    """
    Базовый класс-парсер. Умеет считывать все числа из файла 
    в один плоский список (всеядный парсинг).
    """
    def __init__(self, path):
        self.path = path
        self.nums = self._read_all_numbers()
        if not self.nums:
            raise ValueError(f"Файл {path} пустой!")
        self.N = int(self.nums[0]) 

    def _read_all_numbers(self):
        nums = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                for token in line.split():
                    nums.append(float(token))
        return nums

    def parse(self):
        raise NotImplementedError("Этот метод должен быть переопределен в наследниках!")


class SystemReader(BaseMatrixReader):
    """ Наследник для задач, где нужна Матрица A и Вектор b (LU-разложение 1.1) """
    def parse(self):
        A = []
        pos = 1
        for i in range(self.N):
            row = self.nums[pos : pos + self.N]
            A.append(row)
            pos += self.N
        b = self.nums[pos : pos + self.N]
        return self.N, A, b


class SystemEpsReader(BaseMatrixReader):
    """ Наследник для задач, где нужны Матрица A, Вектор b и точность eps (Итерации 1.3) """
    def parse(self):
        A = []
        pos = 1
        for i in range(self.N):
            row = self.nums[pos : pos + self.N]
            A.append(row)
            pos += self.N
        b = self.nums[pos : pos + self.N]
        pos += self.N
        eps = self.nums[pos] if pos < len(self.nums) else 1e-4
        return self.N, A, b, eps


class TridiagonalReader(BaseMatrixReader):
    """ Наследник для метода прогонки (1.2). Нарезает файл строго на диагонали """
    def parse(self):
        pos = 1
        main_diag = self.nums[pos : pos + self.N]
        pos += self.N
        super_diag = self.nums[pos : pos + (self.N - 1)]
        pos += (self.N - 1)
        sub_diag = self.nums[pos : pos + (self.N - 1)]
        pos += (self.N - 1)
        d = self.nums[pos : pos + self.N]
        return self.N, main_diag, super_diag, sub_diag, d


def matrix_multiply(matrix1, matrix2):
    """ Вручную перемножает две матрицы (алгоритм строка на столбец) """
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])
    if cols1 != rows2:
        raise ValueError("Матрицы таких размерностей нельзя перемножить!")
    
    result = [[0.0 for _ in range(cols2)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


def is_matrix_equal(matrix1, matrix2, eps=1e-7):
    """ Проверяет равенство двух матриц с учетом машинной погрешности eps """
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return False
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if abs(matrix1[i][j] - matrix2[i][j]) > eps:
                return False
    return True

def dot(a, b):
    """ Скалярное произведение двух векторов одинаковой длины """
    if len(a) != len(b):
        raise ValueError("Векторы должны быть одинаковой длины!")
    return sum(x * y for x, y in zip(a, b))
