import os


class BaseNonlinearReader:
    """
    Базовый класс для чтения и валидации числовых параметров нелинейных задач.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def _read_all_numbers(self):
        """ Внутренний метод: считывает все токены из текстового файла в виде чисел """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Файл параметров не найден по пути: {self.file_path}")

        numbers = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                for token in line.split():
                    numbers.append(float(token))
        return numbers


class EquationParamReader(BaseNonlinearReader):
    """
    Парсер параметров для одиночного нелинейного уравнения (пункт 2.1).
    Возвращает точность (eps) и вектор начального приближения x0.
    """
    def parse(self):
        nums = self._read_all_numbers()
        if len(nums) < 2:
            raise ValueError(
                f"Ошибка ввода: Файл {self.file_path} должен содержать минимум 2 числа (eps, x0)."
            )
        eps = nums[0]
        x0 = [nums[1]]  # Универсальный интерфейс требует список даже для одной переменной
        return eps, x0


class SystemParamReader(BaseNonlinearReader):
    """
    Парсер параметров для систем нелинейных уравнений (пункт 2.2).
    Возвращает размерность N, точность (eps) и вектор начального приближения X0.
    """
    def parse(self):
        nums = self._read_all_numbers()
        if not nums:
            raise ValueError(f"Ошибка ввода: Файл {self.file_path} пуст.")

        n = int(nums[0])
        if n <= 0:
            raise ValueError("Ошибка ввода: Размерность N должна быть положительной.")

        required_length = 2 + n
        if len(nums) < required_length:
            raise ValueError(
                f"Ошибка ввода: Требуется {required_length} чисел для системы размера {n}."
            )

        eps = nums[1]
        x0 = nums[2:required_length]
        return n, eps, x0
