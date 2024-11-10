# CircularEmbedding

Investigating embeddings of points in high-dimensional spaces (LLM token encodings) onto a circle.

## Setup

Install dependencies:

```bash
virtualenv venv -P python3.13
source venv/bin/activate
pip install -r requirements.txt
```

## Using the demo code

To see the list of available visualization options:

```bash
python generate_test_vector_json.py --help
```

To visualize one of the test vector sets (requires a lot of
memory for large sets):

```bash
python generate_test_vector_json.py torus > torus.json
python reduce_to_3d.py < torus.json > torus_3d.json
python gif_visualze.py < torus_3d.json > torus.gif
```
