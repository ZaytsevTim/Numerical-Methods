"""Утилиты вывода: таблицы."""
from typing import List, Sequence


def print_table(headers: Sequence[str], rows: List[Sequence], col_width: int = 14) -> None:
    """Печатает таблицу с выравниванием по колонкам."""
    # Верхняя граница
    line = "+".join("-" * (col_width + 2) for _ in headers)
    print("+" + line + "+")

    # Шапка
    header_line = "|".join(f" {h:^{col_width}} " for h in headers)
    print("|" + header_line + "|")
    print("+" + line + "+")

    # Строки
    for row in rows:
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f" {v:^{col_width}.6g} ")
            else:
                cells.append(f" {str(v):^{col_width}} ")
        print("|" + "|".join(cells) + "|")
    print("+" + line + "+")
