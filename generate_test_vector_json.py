"""
Generate JSON files with test vectors corresponding to various geometric shapes.

These vectors can be used as input to the dimensionality reduction algorithms,
whose output can be used in the visualization code.

Usage:
    python generate_test_vector_json.py torus > torus.json
    python generate_test_vector_json.py torus blobs > torus_and_blobs.json
    python generate_test_vector_json.py all > all_shapes.json

Example JSON output (but for 3D vectors, not 768-D vectors):
{
    "points": [
        {
           "point": [0.0, 0.0, 0.0],
           "category": "1"
        },
        {
           "point": [0.1, 0.2, 0.3],
           "category": "1"
        },
        {
           "point": [0.3, 0.2, 0.1],
           "category": "2"
        }
    ]
}
"""

import json
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Shape(Enum):
    """The shapes to generate."""

    def __new__(cls, value: str, description: str):
        """Create a new enum member"""
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    torus = (
        "torus",
        "In the first three dimensions of the 768-dimensional "
        "space, these are 10,000 points located within a solid "
        "torus with an inner radius of 100 and an outer radius "
        "of 120 centered at the origin.",
    )
    cubes = (
        "cubes",
        "In the second three dimensions of the 768-dimensional "
        "space, these 8,000 points are divided into 8 solid "
        "cubes. Each cube has a side length of 10 and is "
        "centered at one of the 8 corners of an 180x180x180 "
        "imaginary cube centered at the origin, that is, "
        "one cube is centered at "
        "(0,0,0, 90,-90,-90,0,...), another at "
        "(0,0,0, 90, 90,-90,0,...), and so on.",
    )
    blobs = (
        "blobs",
        "The third group is 20,000 points in 20 isolated "
        "clusters of 1,000 points each. Each cluster is a "
        "Gaussian cloud with a standard deviation of 2.5 "
        "located at a vertex of a dodecahedron with a radius "
        "of 70 centered at the origin. The blobs extend over"
        "all 768 dimensions but the centers are chosen from "
        "a 3-dimensional dodecahedron embedded in the third "
        "three dimensions. So, some example centers are"
        "(0,0,0,0,0,0,70/sqrt(3),70/sqrt(3),70/sqrt(3),0,...), and "
        "(0,0,0,0,0,0,-70/sqrt(3),70/sqrt(3),70/sqrt(3),0,...).",
    )
    little_sin = (
        "little_sin",
        "The fourth group is 1,000 points in a 3-dimensional "
        "S-curve embedded in the 768-dimensional space. The S-curve "
        "has height 50 and width 50 and is centered at the origin. "
        "The S-curve is defined by the equation "
        "y = 25 * sin(pi x/(25)). where -25 <= x <= 25 and the values "
        "at all other dimensions are uniformly sampled between -1 and 1."
        "So, one point might be "
        "(-0.9,-.2,.3,.55,.21,-.2,-.10,-.15,-.12,20, 25*sin(pi*20/25),"
        " .10, -.22,...), "
        "If we make the confusing directions wider, MDS does not pick"
        "up the S-curve structure as well.",
    )
    big_sin = (
        "big_sin",
        "Like little_sin, but the S-curve has height 100 and width 100. Its "
        "x and y coordinates are 12th and 13th in the 768-dimensional space."
        "In addition it has 2000 points and its confusion dims are -5 to 5.",
    )


# Golden ratio
PHI = (1 + 5**0.5) / 2

# From https://en.wikipedia.org/wiki/Regular_dodecahedron
standard_dodecahedron_vertices = [
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [1, 1, -1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
    [0, PHI, 1 / PHI],
    [0, -PHI, 1 / PHI],
    [0, PHI, -1 / PHI],
    [0, -PHI, -1 / PHI],
    [1 / PHI, 0, PHI],
    [-1 / PHI, 0, PHI],
    [1 / PHI, 0, -PHI],
    [-1 / PHI, 0, -PHI],
    [PHI, 1 / PHI, 0],
    [-PHI, 1 / PHI, 0],
    [PHI, -1 / PHI, 0],
    [-PHI, -1 / PHI, 0],
]

# Distance from the center of the dodecahedron to a vertex
# (1^2 + 1^2 + 1^2) ** 0.5 = (3) ** 0.5
dodecahedron_vertex_to_center = 3**0.5

# Scale the dodecahedron vertices to have a radius of 70
scaled_dodecahedron_vertices = [
    [70 * x / dodecahedron_vertex_to_center for x in vertex]
    for vertex in standard_dodecahedron_vertices
]


@dataclass
class Args:
    """Parsed command-line arguments."""

    shapes: list[Shape]
    seed: int


def parse_args(args: list[str]) -> Args:
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shapes",
        nargs="+",
        type=str,
        choices=[*(s.value for s in list(Shape)), "all"],
        help="The shapes to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The random seed to use."
    )
    a = parser.parse_args(args)
    if "all" in a.shapes:
        a.shapes = list(Shape)
    else:
        a.shapes = [Shape[s] for s in a.shapes]

    return Args(shapes=a.shapes, seed=a.seed)


def torus_points() -> list[np.ndarray]:
    """Return points for a torus"""
    points = []
    inner = 100
    outer = 120
    diff_r = (outer - inner) / 2
    for _ in range(10000):

        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        x = (inner + diff_r * np.cos(theta)) * np.cos(phi)
        y = (inner + diff_r * np.cos(theta)) * np.sin(phi)
        z = diff_r * np.sin(theta)

        point = np.array([x, y, z, *([0] * 765)])
        points.append(point)
    return points


def cubes_points() -> list[np.ndarray]:
    """Return points for cubes"""
    points = []
    for x in [-90, 90]:
        for y in [-90, 90]:
            for z in [-90, 90]:
                template = np.zeros((1000, 768))
                cube = np.random.uniform(-5, 5, size=(1000, 3))
                cube[:, 0:3] += [x, y, z]
                template[:, 3:6] = cube
                points.extend(template)
    return points


def blobs_points() -> list[np.ndarray]:
    """Return points for blobs"""
    points = []
    for vertex_3d in scaled_dodecahedron_vertices:
        vertex = np.zeros((1000, 768))
        vertex[:, 6:9] = vertex_3d
        cluster = np.random.normal(loc=vertex, scale=2.5, size=(1000, 768))
        points.extend(cluster)
    return points


def little_sin_points() -> list[np.ndarray]:
    """Return points for little_sin"""
    points = []
    for _ in range(1000):
        x = np.random.uniform(-25, 25)
        y = 25 * np.sin(np.pi * x / 25)
        before_dims = np.random.uniform(-1, 1, 9)
        other_dims = np.random.uniform(-1, 1, 768 - 2 - 9)
        point = np.array([*before_dims.tolist(), x, y, *other_dims.tolist()])
        points.append(point)
    return points


def big_sin_points() -> list[np.ndarray]:
    """Return points for big_sin"""
    points = []
    for _ in range(2000):
        x = np.random.uniform(-100, 100)
        y = 100 * np.sin(np.pi * x / 100)
        before_dims = np.random.uniform(-5, 5, 11)
        other_dims = np.random.uniform(-5, 5, 768 - 2 - 11)
        point = np.array([*before_dims.tolist(), x, y, *other_dims.tolist()])
        points.append(point)
    return points


POINT_GENERATORS = {
    Shape.torus: torus_points,
    Shape.cubes: cubes_points,
    Shape.blobs: blobs_points,
    Shape.little_sin: little_sin_points,
    Shape.big_sin: big_sin_points,
}


def shape_points(shape: Shape) -> list[np.ndarray]:
    """Return points for the given shape."""
    if shape in POINT_GENERATORS:
        return POINT_GENERATORS[shape]()
    raise ValueError(f"Unknown shape: {shape}")


def main():
    """Generate test vectors and output them as JSON."""
    import sys

    args = parse_args(sys.argv[1:])
    random.seed(args.seed)
    all_points = {shape: shape_points(shape) for shape in args.shapes}

    output = {
        "points": [
            {"point": point.tolist(), "category": shape.name}
            for shape, points in all_points.items()
            for point in points
        ]
    }
    print(json.dumps(output, indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
