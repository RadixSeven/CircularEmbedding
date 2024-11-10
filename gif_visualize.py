"""Read the output of reduce_to_3d.py and visualize it as an animated GIF.

The GIF will have a camera rotating around the 3D points. The camera should
be far enough away from the points to see the entire point cloud.

The visualized points will be colored according to their category string.
Each unique category string will be assigned a unique color. The colors are
chosen from the Bang Wong color palette.

Usage:
    python gif_visualize.py < torus_3d.json > torus.gif
"""

import json
import logging
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Bang Wong color palette
COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


def color_map_for(categories: list[str]) -> dict[str, str]:
    """Return a color map for the categories

    map[category] = HTML hex color string
    """
    unique_categories = sorted(set(categories))
    if len(unique_categories) > len(COLORS):
        logging.warning(
            "More categories (%d) than colors (%d). Some colors will be reused.",
            len(unique_categories),
            len(COLORS),
        )
    unique_categories = sorted(set(unique_categories))
    return {
        category: COLORS[i % len(COLORS)]
        for i, category in enumerate(unique_categories)
    }


def update(frame, scatter, ax):
    """Update the scatter plot for the next frame of the animation."""
    ax.view_init(elev=10.0, azim=frame)
    return (scatter,)


def main():
    """Read JSON input, create a color map, and save an animated GIF."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Reading JSON input")
    input_data = json.load(sys.stdin)
    points = np.array([elt["3d"] for elt in input_data["points"]])
    categories = [elt["category"] for elt in input_data["points"]]
    # noinspection PyUnusedLocal
    input_data = None  # Free up memory

    logging.info("Creating color map")
    color_map = color_map_for(categories)
    colors = [color_map[category] for category in categories]

    logging.info("Creating animated GIF")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=colors, s=1
    )

    ani = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, 360, 2),
        fargs=(scatter, ax),
        interval=50,
    )
    with tempfile.NamedTemporaryFile(
        delete=True, delete_on_close=True, suffix=".gif"
    ) as f:
        logging.info("Saving GIF to %s", f.name)
        ani.save(f.name, writer="pillow", fps=20)
        logging.info("Copying GIF to stdout")
        f.seek(0)
        sys.stdout.buffer.write(f.read())
    logging.info("Done")


if __name__ == "__main__":
    main()
