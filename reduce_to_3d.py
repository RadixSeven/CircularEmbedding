"""Take the output of generate_test_vector_json.py and reduce it to 3D.

Currently, uses Isomap to reduce the dimensionality of the points to 3D. This
does not preserve global structure when the points form good clusters so their
nearest neighbors are all in the same cluster. However, it is much faster and
more memory-efficient than code that requires the full distance matrix.

Usage:
    python reduce_to_3d.py < torus.json > torus_3d.json


Example JSON output (but for 4D vectors, not 768-D vectors):
{
    "points": [
        {
           "point": [0.0, 0.0, 0.0, 0.0],
           "3d": [0.0, 0.0, 0.0],
           "category": "1"
        },
        {
           "point": [0.1, 0.2, 0.3, 0.4],
           "3d": [1.0, 1.0, 1.0],
           "category": "1"
        },
        {
           "point": [0.3, 0.2, 0.1, 0.0],
           "3d": [-1.0, -1.0, -1.0],
           "category": "2"
        }
    ]
}
"""

import json
import logging
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.manifold import Isomap


@dataclass
class Args:
    """Command-line arguments"""

    num_threads: int


def parse_args(args: list[str]) -> Args:
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-threads",
        "-t",
        type=int,
        default=3,
        help="The number of threads to use for reduction.",
    )
    a = parser.parse_args(args)
    return Args(num_threads=a.num_threads)


def reduce_to_3d(points: np.ndarray, num_threads: int | None) -> np.ndarray:
    """Reduce the dimensionality of the points to 3D using Isomap.

    Args:
        points: An array of shape (n_points, n_dimensions) containing the
            original points.
        num_threads: The number of threads to use for MDS. If None, use the
            default number of threads.
    """
    mds = Isomap(n_components=3, metric="euclidean", n_jobs=num_threads)
    return mds.fit_transform(points)


def main():
    """Read JSON input, reduce dimensionality, and output JSON."""
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    input_data = json.load(sys.stdin)
    points = np.array([elt["point"] for elt in input_data["points"]])
    categories = [elt["category"] for elt in input_data["points"]]
    # noinspection PyUnusedLocal
    input_data = None  # Free up memory
    logging.info("Read %d points", len(categories))
    points_3d = reduce_to_3d(points, args.num_threads)

    output_data = {
        "points": [
            {
                "point": points[i].tolist(),
                "3d": points_3d[i].tolist(),
                "category": category,
            }
            for i, category in enumerate(categories)
        ]
    }

    print(json.dumps(output_data, indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
