import math

class JacobiRotationSolver:
    def __init__(self, A, eps):
        """
        Конструктор класса.
        A: исходная симметричная матрица (список списков float)
        eps: заданная точность останова (float)
        """
        self.eps = float(eps)
        self.N = len(A)
        
        # Создаем копию матрицы для последующей диагонализации
        self.A = [[float(x) for x in row] for row in A]
        
        # Инициализируем матрицу собственных векторов единичной матрицей
        self.V = [[1.0 if i == j else 0.0 for j in range(self.N)] for i in range(self.N)]
        
        self.iterations = 0

    def _find_max_off_diagonal(self):
        """
        Поиск максимального по модулю внедиагонального элемента.
        Возвращает значение элемента и его индексы (p, q).
        """
        max_val = 0.0
        p, q = 0, 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if abs(self.A[i][j]) > max_val:
                    max_val = abs(self.A[i][j])
                    p, q = i, j
        return max_val, p, q

    def _calc_rotation_angle(self, p, q):
        """
        Расчет значений тригонометрических функций угла поворота.
        Предусмотрена обработка случая равенства диагональных элементов.
        """
        if abs(self.A[p][p] - self.A[q][q]) < 1e-9:
            phi = math.pi / 4
            return math.cos(phi), math.sin(phi)
        
        tan_2phi = (2.0 * self.A[p][q]) / (self.A[p][p] - self.A[q][q])
        phi = 0.5 * math.atan(tan_2phi)
        return math.cos(phi), math.sin(phi)

    def solve(self, max_iterations=5000):
        """
        Выполнение итерационного процесса вращений.
        Возвращает список собственных значений, матрицу собственных векторов и число итераций.
        """
        while self.iterations < max_iterations:
            max_val, p, q = self._find_max_off_diagonal()
            
            # Критерий останова по достижении заданной точности внедиагональных элементов
            if max_val < self.eps:
                lambdas = [self.A[i][i] for i in range(self.N)]
                return lambdas, self.V, self.iterations
                
            self.iterations += 1
            c, s = self._calc_rotation_angle(p, q)
            
            # Локальное обновление элементов матрицы A (двусторонний поворот)
            A_new = [[element for element in row] for row in self.A]
            
            for i in range(self.N):
                if i != p and i != q:
                    A_new[i][p] = c * self.A[i][p] + s * self.A[i][q]
                    A_new[p][i] = A_new[i][p]
                    
                    A_new[i][q] = -s * self.A[i][p] + c * self.A[i][q]
                    A_new[q][i] = A_new[i][q]
            
            A_new[p][p] = c*c * self.A[p][p] + 2.0 * c * s * self.A[p][q] + s*s * self.A[q][q]
            A_new[q][q] = s*s * self.A[p][p] - 2.0 * c * s * self.A[p][q] + c*c * self.A[q][q]
            A_new[p][q] = 0.0
            A_new[q][p] = 0.0
            
            self.A = A_new
            
            # Накопление матрицы собственных векторов (умножение матрицы V на матрицу вращения справа)
            for i in range(self.N):
                v_ip = self.V[i][p]
                v_iq = self.V[i][q]
                self.V[i][p] = c * v_ip + s * v_iq
                self.V[i][q] = -s * v_ip + c * v_iq
                
        raise ValueError(f"Метод вращений не сошелся за {max_iterations} итераций.")
