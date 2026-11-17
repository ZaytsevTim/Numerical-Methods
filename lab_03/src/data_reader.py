"""Чтение числовых данных из текстовых файлов."""
import os
from typing import List, Tuple, Optional


class DataReader:
    """Базовый класс для чтения числовых данных из текстовых файлов."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._read_all_numbers()
        if not self.data:
            raise ValueError(f"Файл {file_path} пуст!")

    def _read_all_numbers(self) -> List[float]:
        nums = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Пропускаем пустые строки и комментарии
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for token in stripped.split():
                    nums.append(float(token))
        return nums


class InterpolationData(DataReader):
    """Формат: N, x[0..N-1], y[0..N-1], x*."""

    def parse(self) -> Tuple[List[float], List[float], float]:
        n = int(self.data[0])
        if len(self.data) < 2 * n + 2:
            raise ValueError(
                f"Ожидалось {2 * n + 2} чисел, получено {len(self.data)}"
            )
        x = self.data[1:n + 1]
        y = self.data[n + 1:2 * n + 1]
        x_star = self.data[2 * n + 1]
        return x, y, x_star


class SplineData(DataReader):
    """Формат: N, x[0..N-1], y[0..N-1], x*."""

    def parse(self) -> Tuple[List[float], List[float], float]:
        n = int(self.data[0])
        if len(self.data) < 2 * n + 2:
            raise ValueError(
                f"Ожидалось {2 * n + 2} чисел, получено {len(self.data)}"
            )
        x = self.data[1:n + 1]
        y = self.data[n + 1:2 * n + 1]
        x_star = self.data[2 * n + 1]
        return x, y, x_star


class LeastSquaresData(DataReader):
    """Формат: N, x[0..N-1], y[0..N-1]."""

    def parse(self) -> Tuple[List[float], List[float]]:
        n = int(self.data[0])
        if len(self.data) < 2 * n + 1:
            raise ValueError(
                f"Ожидалось {2 * n + 1} чисел, получено {len(self.data)}"
            )
        x = self.data[1:n + 1]
        y = self.data[n + 1:2 * n + 1]
        return x, y


class DifferentiationData(DataReader):
    """Формат: N, x[0..N-1], y[0..N-1], x*."""

    def parse(self) -> Tuple[List[float], List[float], float]:
        n = int(self.data[0])
        if len(self.data) < 2 * n + 2:
            raise ValueError(
                f"Ожидалось {2 * n + 2} чисел, получено {len(self.data)}"
            )
        x = self.data[1:n + 1]
        y = self.data[n + 1:2 * n + 1]
        x_star = self.data[2 * n + 1]
        return x, y, x_star


class IntegrationData(DataReader):
    """Формат: a, b, h1, [h2], [exact]."""

    def parse(self) -> Tuple[float, float, float, float, Optional[float]]:
        a = self.data[0]
        b = self.data[1]
        h1 = self.data[2]
        h2 = self.data[3] if len(self.data) > 3 else h1 / 2
        exact = self.data[4] if len(self.data) > 4 else None
        return a, b, h1, h2, exact


# ---------- Фабрики ----------
def _path(data_dir: str, filename: str) -> str:
    return os.path.join(data_dir, filename)


def read_interpolation_data(data_dir: str, filename: str = "interpolation.txt") -> InterpolationData:
    return InterpolationData(_path(data_dir, filename))


def read_spline_data(data_dir: str, filename: str = "spline.txt") -> SplineData:
    return SplineData(_path(data_dir, filename))


def read_least_squares_data(data_dir: str, filename: str = "least_squares.txt") -> LeastSquaresData:
    return LeastSquaresData(_path(data_dir, filename))


def read_differentiation_data(data_dir: str, filename: str = "differentiation.txt") -> DifferentiationData:
    return DifferentiationData(_path(data_dir, filename))


def read_integration_data(data_dir: str, filename: str = "integration.txt") -> IntegrationData:
    return IntegrationData(_path(data_dir, filename))
