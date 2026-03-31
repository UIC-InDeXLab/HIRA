# Methods in method_comparison_bench.py

## Clustering Methods

- **k-means**: Standard Lloyd's algorithm that iteratively assigns keys to the nearest center and recomputes centers as cluster means, minimizing within-cluster Euclidean distance.

- **Spherical k-means**: Runs k-means on L2-normalized keys so that clustering optimizes for angular similarity (cosine distance) rather than Euclidean distance.

- **Random Projection**: Projects keys onto random unit vectors, binarizes by median to form hash codes, and maps hashes to buckets. A final nearest-center reassignment refines the clusters.

- **Random Partition**: Baseline that assigns keys to clusters uniformly at random with no structure, used to measure how much real clustering helps pruning.

- **PQ Subspace**: Splits the key dimensions into subspaces, runs mini k-means on each independently, then combines subspace assignments via a composite hash to form the final clustering.

- **Diagonal GMM**: Fits a Gaussian mixture model with diagonal covariances using EM (initialized from k-means). Assigns each key to the component with highest responsibility, capturing per-dimension variance unlike k-means.

## Enclosing Methods

- **Ball (Centroid)**: Places a ball at the k-means centroid with radius equal to the maximum distance from the centroid to any child in the cluster. The gate checks if `center . q + radius > threshold`.

- **Minimum Enclosing Ball**: Iteratively shifts the ball center toward the farthest point in each cluster to approximate a tighter minimum enclosing ball, typically yielding smaller radii than the centroid ball.

- **AABB (Axis-Aligned Bounding Box)**: Computes per-dimension min/max bounds for each cluster. The gate computes the maximum possible dot product with the query by picking the best corner of the box per dimension: `sum_d max(q_d * lo_d, q_d * hi_d)`.

- **Cone**: Defines a cone per cluster using the mean direction of normalized children and a half-angle covering all children. The gate upper-bounds `q . k` for any key inside the cone using the angular gap between the query and cone axis.
