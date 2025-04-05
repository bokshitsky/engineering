import argparse
import csv
import random
import sys


def main():
    args = _parse_args()
    column_specs = [s.strip() for s in args.columns.split(",")]
    parsed_columns = [_parse_column_spec(spec) for spec in column_specs]

    column_names = [name for name, _, _ in parsed_columns]
    if column_names[0] != "id":
        column_names.insert(0, "id")

    writer = csv.writer(sys.stdout)
    writer.writerow(column_names)
    for i in range(args.rows):
        row = [str(i + 1)]
        row.extend(
            [round(random.uniform(low, high), 2) for _, low, high in parsed_columns]
        )
        writer.writerow(row)


def _parse_args():
    parser = argparse.ArgumentParser(description="Генерация CSV с случайными числами.")
    parser.add_argument(
        "--columns",
        type=str,
        required=True,
        help="Список определение столбцов через запятую, например: Mx:100:500,My:100,Pz",
    )
    parser.add_argument(
        "--rows",
        type=int,
        required=True,
        help="Количество строк, которые нужно сгенерировать.",
    )
    return parser.parse_args()


def _parse_column_spec(column_spec: str) -> tuple[str, float, float]:
    column_def = column_spec.split(":")
    if len(column_def) > 3:
        raise ValueError(f"Неверный формат столбца: {column_spec}")

    name = column_def[0]
    if len(column_def) == 1:
        return name, -500.0, 500.0

    if len(column_def) == 2:
        value = float(column_def[1])
        return name, -value, value

    return name, float(column_def[1]), float(column_def[2])


if __name__ == "__main__":
    main()
