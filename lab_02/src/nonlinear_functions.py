import math

# Константа 'a' для Варианта 10 из таблицы
A_PARAM = 1.0

# ==============================================================================
# ПУНКТ 2.1: ОДИНОЧНОЕ УРАВНЕНИЕ (f(x) = sin(x) - 2x^2 + 0.5)
# ==============================================================================
def single_equation_f(x_vector):
    """ Исходная функция f(x) = sin(x) - 2x^2 + 0.5 """
    x = x_vector[0]
    return [math.sin(x) - 2.0 * (x**2) + 0.5]

def single_equation_jacobian(x_vector):
    """ Производная функции f'(x) = cos(x) - 4x в виде матрицы 1x1 """
    x = x_vector[0]
    return [[math.cos(x) - 4.0 * x]]

def single_equation_phi(x_vector):
    """ Функция шага итераций МПИ: phi(x) = sqrt(0.5 * sin(x) + 0.25) """
    x = x_vector[0]
    return [math.sqrt(0.5 * math.sin(x) + 0.25)]

def single_equation_domain_check(x_vector):
    """ Проверка области определения: подкоренное выражение должно быть >= 0 """
    x = x_vector[0]
    return not (0.5 * math.sin(x) + 0.25 < 0.0)


# ==============================================================================
# ПУНКТ 2.2: НЕЛИНЕЙНАЯ СИСТЕМА (Вариант 10)
# ==============================================================================
def nonlinear_system_f(x_vector):
    """ Вектор невязок нелинейной системы F(X) """
    x1 = x_vector[0]
    x2 = x_vector[1]

    f1 = x1 - math.cos(x2) - A_PARAM
    f2 = x2 - math.sin(x1) - A_PARAM
    return [f1, f2]

def nonlinear_system_jacobian(x_vector):
    """ Матрица Якоби J(X) для аналитических частных производных """
    x1 = x_vector[0]
    x2 = x_vector[1]

    return [
        [1.0, math.sin(x2)],
        [-math.cos(x1), 1.0]
    ]

def nonlinear_system_phi(x_vector):
    """ Вектор-функция шага итераций МПИ: X = Phi(X) """
    x1 = x_vector[0]
    x2 = x_vector[1]

    x1_next = A_PARAM + math.cos(x2)
    x2_next = A_PARAM + math.sin(x1)
    return [x1_next, x2_next]

def nonlinear_system_domain_check(x_vector):
    """ Область определения системы непрерывна на всем вещественном пространстве """
    return True
