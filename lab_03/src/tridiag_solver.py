class TridiagonalSolver:
    def __init__(self, main_diag, super_diag, sub_diag, d):
        """
        Конструктор: принимает 4 плоские ленты чисел из парсера.
        Инкапсулирует состояние и выделяет память под массивы alpha, beta и решения x.
        """
        self.B = [float(x) for x in main_diag]  # Главная диагональ (длина N)
        self.C = [float(x) for x in super_diag] # Верхняя побочная диагональ (длина N-1)
        self.A = [float(x) for x in sub_diag]   # Нижняя побочная диагональ (длина N-1)
        self.D = [float(x) for x in d]          # Вектор свободных членов справа (длина N)
        
        self.N = len(self.B)
        
        # Выделяем память под прогоночные коэффициенты (длина N + 1 для удобства индексов)
        self.alpha = [0.0 for _ in range(self.N + 1)]
        self.beta = [0.0 for _ in range(self.N + 1)]
        self.x = [0.0 for _ in range(self.N)]

    def check_diagonal_dominance(self):
        """
        Метод проверки устойчивости прогонки (диагональное преобладание).
        """
        has_strict = False
        for i in range(self.N):
            current_b = abs(self.B[i])
            
            # Сосед слева (нижняя диагональ A)
            current_a = abs(self.A[i - 1]) if i > 0 else 0.0
            # Сосед справа (верхняя диагональ C)
            current_c = abs(self.C[i]) if i < self.N - 1 else 0.0
            
            if current_b < (current_a + current_c):
                print(f"[Предупреждение] На строке {i} нарушено диагональное преобладание!")
                return False
            if current_b > (current_a + current_c):
                has_strict = True
        return has_strict

    def _forward_step(self):
        """
        Внутренний метод: Прямой ход прогонки.
        Идем сверху вниз и вычисляем коэффициенты-шпаргалки alpha и beta.
        """
        # Стартовая позиция для первой строки матрицы (соседа слева A еще нет)
        if abs(self.B[0]) < 1e-9:
            raise ValueError("Деление на ноль на старте прогонки! Нулевой элемент на диагонали.")
            
        self.alpha[1] = -self.C[0] / self.B[0]
        self.beta[1] = self.D[0] / self.B[0]
        
        # Движение по конвейеру вниз для промежуточных строк
        for i in range(1, self.N - 1):
            denominator = self.B[i] + self.A[i - 1] * self.alpha[i]
            if abs(denominator) < 1e-9:
                raise ValueError(f"Деление на ноль на шаге {i} в прямой прогонке!")
                
            self.alpha[i + 1] = -self.C[i] / denominator
            self.beta[i + 1] = (self.D[i] - self.A[i - 1] * self.beta[i]) / denominator
            
        # Финальный шаг на самом дне (последняя строка, соседа справа C больше нет)
        denominator = self.B[self.N - 1] + self.A[self.N - 2] * self.alpha[self.N - 1]
        if abs(denominator) < 1e-9:
            raise ValueError("Деление на ноль на последнем шаге прямой прогонки!")
            
        self.beta[self.N] = (self.D[self.N - 1] - self.A[self.N - 2] * self.beta[self.N - 1]) / denominator

    def _backward_step(self):
        """
        Внутренний метод: Обратный ход прогонки.
        Разворачиваемся на дне и распутываем иксы снизу вверх по шпаргалкам.
        """
        # Самый нижний икс равен последней бете
        self.x[self.N - 1] = self.beta[self.N]
        
        # Поднимаемся обратно к первой строке
        for i in range(self.N - 2, -1, -1):
            self.x[i] = self.alpha[i + 1] * self.x[i + 1] + self.beta[i + 1]

    def solve(self):
        """
        Публичный интерфейс: запускает весь конвейер и возвращает историю для ТЗ.
        """
        self.check_diagonal_dominance()
        self._forward_step()
        self._backward_step()
        return self.alpha, self.beta, self.x
