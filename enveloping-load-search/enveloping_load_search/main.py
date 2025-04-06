import argparse
import itertools
import sys
from enum import Enum, auto
from typing import Iterable

import pandas
import pandas as pd
import plotly.express as px


class Direction(Enum):
    POS = auto()
    NEG = auto()

    def compare(self, v1: float, v2: float) -> bool:
        if self == Direction.POS:
            return v2 > v1
        return v2 < v1


def is_more_critical(point1: list[float], point2: list[float], directions: Iterable[Direction]) -> bool:
    return all([d.compare(v1, v2) for v1, v2, d in zip(point1, point2, directions)])


def main():
    parser = argparse.ArgumentParser(description="Build load graph")
    parser.add_argument("filepath", help=".csv file path")
    args = parser.parse_args()

    df = pd.read_csv(args.filepath)
    columns = list(df.columns)[1:]

    # All possible directions of all dimensions
    all_directions = list(itertools.product([Direction.POS, Direction.NEG], repeat=len(columns)))

    # Iterate over all points
    values = df.drop("id", axis=1)
    for i, row_i in values.iterrows():
        point1 = list(row_i)
        has_more_critical_point_by_direction = {d: False for d in all_directions}

        # Iterate over all points inside
        for j, row_j in values.iterrows():
            if i == j:
                continue
            point2 = list(row_j)

            # Iterate over all directions and check if point2 is more critical than point1
            for direction in all_directions:
                if not has_more_critical_point_by_direction[direction] and is_more_critical(point1, point2, direction):
                    has_more_critical_point_by_direction[direction] = True

            # Assume that point is not critical, if for all directions where is at least one more critical point
            if all(has_more_critical_point_by_direction.values()):
                break

        df.at[i, "is_enveloping"] = not all(has_more_critical_point_by_direction.values())

    _save_results(args.filepath, columns, df)


def _save_results(csv_filename: str, columns: list[str], df: pandas.DataFrame):
    fig_function = px.scatter_3d if len(columns) == 3 else px.scatter
    scatter_args = dict((zip(("x", "y", "z"), columns)))
    fig = fig_function(
        df,
        hover_data=["id"],
        color="is_enveloping",
        color_discrete_map={
            True: "red",
            False: "blue",
        },
        **scatter_args,
    )
    result_file = csv_filename + ".enveloping.html"
    fig.write_html(result_file)
    print(f"saved {result_file}")
    df.to_csv(sys.stdout, index=False)
    fig.show()


if __name__ == "__main__":
    main()
