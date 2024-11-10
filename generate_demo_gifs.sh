#!/usr/bin/env bash
# Note that we don't need to process the case when there is only one shape: a
# repeated shape is the same as a single shape.
for first_shape in torus cubes blobs little_sin big_sin; do
  for second_shape in torus cubes blobs little_sin big_sin; do
    # Assign f to the alphabetically first shape and s to the second shape.
    if [ "$first_shape" \> "$second_shape" ]; then
      f=$second_shape
      s=$first_shape
    else
      f=$first_shape
      s=$second_shape
    fi
    both="${f}_${s}"
    if [ -e $both.gif ]; then
      echo "Skipping $both because it already exists."
      continue
    fi
    echo "Processing $both"
    nice python generate_test_vector_json.py $f $s \
        > $both.json && \
        nice python reduce_to_3d.py -t "-1" < $both.json > ${both}_3d.json && \
        nice python gif_visualize.py < ${both}_3d.json > $both.gif
  done
done
echo "Processing all_shapes"
nice python generate_test_vector_json.py all \
    > all_shapes.json && \
nice python reduce_to_3d.py -t "-1" \
    < all_shapes.json > all_shapes_3d.json && \
nice python gif_visualize.py < all_shapes_3d.json > all_shapes.gif;