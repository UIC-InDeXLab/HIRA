# Methods in `benchmark_area/quick_pruning`

## Clustering Methods

- **k-means**: Standard Lloyd updates in the original key space, minimizing within-cluster L2 distortion.

- **Spherical k-means**: Runs k-means on L2-normalized keys so assignments follow angular similarity rather than Euclidean distance.

- **Random Projection**: Uses random projections and median thresholding to form hash buckets, then refines with nearest-center reassignment.

- **Random Partition**: Assigns keys to clusters uniformly at random. Useful as a lower-bound baseline for how much structure matters.

- **PQ Subspace**: Splits dimensions into subspaces, clusters each subspace independently, then combines the subspace codes into a composite cluster id.

- **PQ L2**: Uses PQ-style subspace clustering for initialization, then refines with ordinary L2 k-means.

- **PQ Linf**: PQ-style initialization followed by Linf-oriented refinement to reduce worst-coordinate spread.

- **PQ8**: PQ-style clustering with a fixed 8-way subspace coding scheme.

- **PQ Span**: Starts from PQ-subspace assignments, then refines by minimizing how much each point expands the destination cluster's AABB.

- **Whitened PQ**: Normalizes each dimension by global per-head standard deviation before PQ clustering, then maps centers back to the original space. This reduces domination by a few high-variance coordinates and worked well with AABB gating.

- **Whitened PQ Span**: Combines whitened PQ initialization with span-aware refinement. In the recent experiments this was the strongest clustering method for `aabb`.

- **Interleaved Whitened PQ**: Whitens dimensions, sorts them by variance, then distributes them round-robin across PQ subspaces so each subspace sees a mix of strong and weak coordinates.

- **Diagonal GMM**: Fits a diagonal-covariance Gaussian mixture with EM and assigns each point to the highest-responsibility component.

- **k-center**: Greedy farthest-point seeding followed by centroid refinement, targeting tight ball radii.

- **k-means++**: Standard k-means with distance-aware seeding to reduce bad initializations.

- **KD-tree**: Recursively splits along the widest axis at the median, producing balanced, box-friendly partitions.

- **PCA Tree**: Like KD-tree, but first projects into a small PCA basis and then splits along the widest projected axis.

- **PCA k-means**: Runs Lloyd iterations in a low-rank PCA subspace, then computes final centers in the original space.

- **PCA Bisect**: Recursive PCA-guided bisection intended to make elongated clusters easier to separate.

- **Linf k-means**: A k-means-style clustering objective based on coordinatewise worst-case deviation instead of only L2 distance.

- **Span k-means**: Assigns each point to the cluster whose AABB it enlarges the least, directly targeting `aabb`-style pruning quality.

## Enclosing Methods

- **Ball (Centroid)**: Uses the centroid as center and stores the farthest child distance as radius. The gate is `center . q + radius > threshold`.

- **Minimum Enclosing Ball**: Approximates a tighter enclosing ball by iteratively shifting toward farthest points.

- **AABB (Axis-Aligned Bounding Box)**: Stores per-dimension min/max values and upper-bounds the dot product by choosing the best corner for each coordinate.

- **Cone**: Uses a mean direction plus a half-angle and max norm to upper-bound support in angular space.

- **Hybrid**: Intersects several cheap bounds such as ball, AABB, and cone to tighten pruning.

- **Ellipsoid**: Uses a diagonal ellipsoidal scaling around the cluster center to capture anisotropic spread more tightly than a ball.

- **Multi-ball**: Covers a cluster with a small set of balls to handle multimodal structure inside one parent.

- **Subspace Box**: Builds a box in a low-rank local subspace plus a residual norm bound for the orthogonal complement.

- **Tight Hybrid**: Combines several of the strongest complementary bounds, typically ball, box, subspace, and multi-anchor terms.

- **Top-k AABB Residual**: Uses exact AABB support on a few salient dimensions and an L2 residual bound for the remaining ones.

- **Centerline**: Uses the centroid direction as a cheap rank-1 local axis and bounds the remaining orthogonal energy with a residual ball.

- **Dual Centerline**: Extends centerline to two cluster-local axes, giving a tighter rank-2 bound at slightly higher gate cost.

- **PCA OBB**: Rotates keys into a PCA basis and applies an axis-aligned box in that rotated space.

- **Global PCA Box**: Uses a small global PCA basis per head, stores a box in that subspace, and adds a residual norm bound outside the basis.

- **Hybrid Plus**: A larger intersection of ball, AABB, cone, ellipsoid, and centerline style bounds.

- **Split AABB**: Splits each cluster into two sub-boxes and passes if either sub-box can exceed the threshold.

- **Quad AABB**: Generalizes split-style boxing to four sub-boxes for tighter but more expensive support evaluation.

- **Split Hybrid**: Applies split-style partitioning and then intersects with additional bounds such as ball or ellipsoid terms.

- **Bisect AABB**: Uses a two-way AABB split with an additional fallback bound, typically a ball.

- **Slab Bundle**: Projects onto several directions, stores min/max slab extents, and optionally intersects with a ball-style fallback.

- **Split Full Hybrid**: A high-cost intersection of several split and non-split geometric bounds.

- **Outlier AABB**: Builds the box on the cluster core after removing a few farthest points, then keeps those outliers explicitly as point supports. This was useful experimentally but is not yet part of the main enclosing registry.
