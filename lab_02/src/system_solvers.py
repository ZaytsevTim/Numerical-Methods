from lu_decomposition import LUPDecomposition


class UniversalNewtonSolver:
    """
    Универсальный метод Ньютона для систем нелинейных уравнений.
    Работает как с одиночным уравнением (1x1), так и с многомерной системой.
    """
    def __init__(self, f, jacobian, eps, max_iterations=1000, domain_check=None):
        self.f = f
        self.jacobian = jacobian
        self.eps = float(eps)
        self.max_iterations = int(max_iterations)
        self.domain_check = domain_check

    def solve(self, x0):
        """ Запускает итерационный процесс метода Ньютона """
        x = list(x0)
        history = []

        for step in range(self.max_iterations):
            if self.domain_check is not None and not self.domain_check(x):
                raise ValueError(f"Точка {x} выходит за область определения функции.")

            F = self.f(x)
            J = self.jacobian(x)
            N = len(F)

            # Невязка в текущей точке (норма Чебышёва)
            residual = max(abs(fi) for fi in F)

            # Записываем состояние ДО шага (x_k, невязка в x_k)
            history.append({
                "iter": step,
                "x": list(x),
                "residual": residual,
                "step_diff": None,   # шаг появится после обновления
            })

            if residual < self.eps:
                return x, history

            neg_f = [-fi for fi in F]

            # Случай 1x1 (одиночное уравнение)
            if N == 1:
                if abs(J[0][0]) < 1e-12:
                    raise ValueError(f"Нулевая производная при x = {x}")
                delta = [neg_f[0] / J[0][0]]
            else:
                # Случай NxN — вызываем НАШ ЛОКАЛЬНЫЙ LUP-решатель
                solver = LUPDecomposition(J, eps=1e-12)
                delta = solver.solve(neg_f)

            # Обновляем решение: x_new = x + delta
            x_new = [x[i] + delta[i] for i in range(N)]

            # Шаг = max|x_new - x_old|
            step_diff = max(abs(x_new[i] - x[i]) for i in range(N))

            # Дописываем шаг в последнюю запись history (это шаг, сделанный ИЗ x_k)
            history[-1]["step_diff"] = step_diff

            x = x_new

        raise ValueError(f"Метод Ньютона не сошёлся за {self.max_iterations} итераций!")


class UniversalMPISolver:
    """
    Универсальный метод простых итераций для систем нелинейных уравнений.
    Использует переданную функцию шага phi(x).
    """
    def __init__(self, f, phi, eps, max_iterations=10000, domain_check=None):
        self.f = f
        self.phi = phi
        self.eps = float(eps)
        self.max_iterations = int(max_iterations)
        self.domain_check = domain_check

    def solve(self, x0):
        """ Запускает итерационный процесс методом простых итераций """
        x = list(x0)
        history = []

        for step in range(self.max_iterations):
            if self.domain_check is not None and not self.domain_check(x):
                raise ValueError(f"Точка {x} выходит за область определения функции.")

            x_read = list(x)

            # Шаг итерации
            x_new = self.phi(x_read)

            # Шаг = max|x_new - x_old|
            step_diff = max(abs(x_new[k] - x_read[k]) for k in range(len(x_new)))

            # Невязка в новой точке (для согласованности с Ньютоном)
            F_new = self.f(x_new)
            residual = max(abs(fi) for fi in F_new)

            # Записываем НОВУЮ точку x_{k+1} с её невязкой и сделанным шагом
            history.append({
                "iter": step + 1,
                "x": list(x_new),
                "residual": residual,
                "step_diff": step_diff,
            })

            x = x_new

            if step_diff < self.eps:
                return x, history

        raise ValueError(f"Метод простых итераций не сошёлся за {self.max_iterations} итераций!")