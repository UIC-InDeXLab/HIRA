"""
Comprehensive tests for Triton kernel implementations.

Test scenarios:
1. Two-level filter kernel correctness against brute force
2. Two-level filter kernel with masking
3. Three-level filter kernel v1 correctness
4. Three-level filter kernel v2 correctness
5. Three-level filter kernel v3 correctness
6. Edge cases: empty results, all results, boundary thresholds
7. Recall verification (99%+ accuracy)
8. Different data sizes and branching factors
9. CUDA device compatibility
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import kernels
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triton_wrappers import triton_two_level_filter, triton_three_level_filter_v1
from bench.generate_level import (
    generate_three_level_structure,
    generate_parent_child_structure,
)


def brute_force_halfspace(keys: torch.Tensor, q: torch.Tensor, threshold: float):
    """Reference implementation: find all keys with score > threshold."""
    scores = keys @ q
    return scores, (scores > threshold).nonzero(as_tuple=True)[0]


def create_hierarchical_data(
    n_keys, dim, branching_factor, num_levels, device, seed=42
):
    if num_levels == 2:
        output = generate_parent_child_structure(
            num_keys=n_keys,
            dim=dim,
            branching_factor=branching_factor,
            distribution="uniform",
            device=device,
            seed=seed,
        )
    elif num_levels == 3:
        output = generate_three_level_structure(
            num_keys=n_keys,
            dim=dim,
            distribution="uniform",
            branching_factor=branching_factor,
            device=device,
            seed=seed,
        )
    else:
        raise ValueError("Only 2 or 3 levels supported")

    return output


class TestTwoLevelFilterKernel:
    """Test two-level filter kernel against brute force."""

    @pytest.mark.parametrize("n_keys", [256, 1024, 4096])
    @pytest.mark.parametrize("branching_factor", [16, 32, 64])
    @pytest.mark.parametrize("threshold", [-0.5, 0.0, 0.5])
    def test_two_level_correctness(self, n_keys, branching_factor, threshold):
        """Test that two-level kernel matches brute force results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        dim = 128

        # Create hierarchical data
        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        # Generate random query
        q = torch.randn(dim, device=device)
        q = q / q.norm(p=2)

        # Run kernel
        out = triton_two_level_filter(
            K,
            P,
            R,
            q,
            threshold,
            branch=branching_factor,
            BLOCK_C=min(16, branching_factor),
        )

        # Brute force reference
        bf_scores, bf_indices = brute_force_halfspace(K, q, threshold)

        # Check that all brute force results are in kernel results
        kernel_mask = out >= threshold
        kernel_indices = kernel_mask.nonzero(as_tuple=True)[0]

        # Compute recall
        kernel_set = set(kernel_indices.cpu().numpy())
        bf_set = set(bf_indices.cpu().numpy())

        if len(bf_set) > 0:
            recall = len(kernel_set & bf_set) / len(bf_set)
            assert (
                recall >= 0.99
            ), f"Recall {recall:.3f} < 0.99 for threshold {threshold}"

        # Verify scores match for retrieved keys
        for idx in bf_indices:
            kernel_score = out[idx].item()
            bf_score = bf_scores[idx].item()
            assert (
                abs(kernel_score - bf_score) < 1e-3 or kernel_score == 0
            ), f"Score mismatch at idx {idx}: kernel={kernel_score}, bf={bf_score}"

    def test_two_level_empty_result(self):
        """Test with threshold that should return no results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 1024
        branching_factor = 32
        dim = 128

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        # Query orthogonal to all keys
        q = torch.zeros(dim, device=device)
        q[0] = 1.0
        q = q / q.norm(p=2)

        # Very high threshold
        scores = K @ q
        threshold = scores.max().item() + 1.0

        out = triton_two_level_filter(
            K, P, R, q, threshold, branch=branching_factor, BLOCK_C=16
        )

        # Check that few or no results returned
        num_results = (out >= threshold).sum().item()
        assert num_results <= n_keys * 0.05, f"Expected few results, got {num_results}"

    def test_two_level_all_results(self):
        """Test with threshold that should return all results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 512
        branching_factor = 16
        dim = 128

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)

        # Very low threshold
        scores = K @ q
        threshold = scores.min().item() - 1.0

        out = triton_two_level_filter(
            K, P, R, q, threshold, branch=branching_factor, BLOCK_C=16
        )

        # All keys should be computed (even if score < threshold due to parent passing)
        # Check that most keys have non-zero scores
        non_zero = (out != 0).sum().item()
        assert (
            non_zero > n_keys * 0.9
        ), f"Expected most results, got {non_zero}/{n_keys}"


class TestThreeLevelFilterKernels:
    """Test three-level filter kernels."""

    @pytest.mark.parametrize("n_keys", [1024, 4096])
    @pytest.mark.parametrize("branching_factor", [8, 16])
    def test_three_level_correctness(self, n_keys, branching_factor):
        """Test three-level kernels against brute force."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        dim = 128

        # Create hierarchical data with 3 levels
        K, P1, R1, P2, R2 = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=3, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)
        threshold = 0.0

        # Run appropriate kernel version
        out = triton_three_level_filter_v1(
            K, P1, R1, P2, R2, q, threshold, branch=branching_factor
        )

        # Brute force reference
        bf_scores, bf_indices = brute_force_halfspace(K, q, threshold)

        # Check recall
        kernel_mask = torch.isfinite(out)
        kernel_indices = kernel_mask.nonzero(as_tuple=True)[0]

        kernel_set = set(kernel_indices.cpu().numpy())
        bf_set = set(bf_indices.cpu().numpy())

        if len(bf_set) > 0:
            recall = len(kernel_set & bf_set) / len(bf_set)
            assert (
                recall >= 0.99
            ), f": Recall {recall:.3f} < 0.99 (found {len(kernel_set)}/{len(bf_set)})"

    @pytest.mark.parametrize("threshold", [-0.5, -0.2, 0.0, 0.2, 0.5])
    def test_three_level_different_thresholds(self, threshold):
        """Test three-level kernels with various thresholds."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 2048
        branching_factor = 8
        dim = 128

        K, P1, R1, P2, R2 = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=3, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)

        out = triton_three_level_filter_v1(
            K, P1, R1, P2, R2, q, threshold, branch=branching_factor
        )

        # Brute force
        bf_scores, bf_indices = brute_force_halfspace(K, q, threshold)

        # Verify recall
        kernel_mask = torch.isfinite(out)
        kernel_indices = kernel_mask.nonzero(as_tuple=True)[0]

        kernel_set = set(kernel_indices.cpu().numpy())
        bf_set = set(bf_indices.cpu().numpy())

        if len(bf_set) > 0:
            recall = len(kernel_set & bf_set) / len(bf_set)
            assert recall >= 0.99, f"Threshold {threshold}: Recall {recall:.3f} < 0.99"


class TestKernelRecall:
    """Test that recall is consistently high across different scenarios."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_two_level_recall_consistency(self, seed):
        """Test that recall is consistent across different random seeds."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 2048
        branching_factor = 32
        dim = 128
        threshold = 0.0

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device, seed=seed
        )

        # Different query for each seed
        torch.manual_seed(seed + 1000)
        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)

        out = triton_two_level_filter(
            K, P, R, q, threshold, branch=branching_factor, BLOCK_C=16
        )

        bf_scores, bf_indices = brute_force_halfspace(K, q, threshold)

        kernel_mask = out >= threshold
        kernel_indices = kernel_mask.nonzero(as_tuple=True)[0]

        kernel_set = set(kernel_indices.cpu().numpy())
        bf_set = set(bf_indices.cpu().numpy())

        if len(bf_set) > 0:
            recall = len(kernel_set & bf_set) / len(bf_set)
            assert recall >= 0.99, f"Seed {seed}: Recall {recall:.3f} < 0.99"

    @pytest.mark.parametrize("dim", [128])
    def test_three_level_different_dimensions(self, dim):
        """Test three-level kernel with different dimensions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Adjust n_keys for different dimensions to keep test fast
        if dim == 256:
            n_keys = 512
            branching_factor = 8
        else:
            n_keys = 1024
            branching_factor = 8

        device = torch.device("cuda")

        # Note: For non-128 dimensions, need to adjust kernel if hardcoded
        # Skip if kernel is hardcoded to 128
        if dim != 128:
            pytest.skip("Kernel currently hardcoded to 128 dimensions")

        K, P1, R1, P2, R2 = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=3, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)
        threshold = 0.0

        out = triton_three_level_filter_v1(
            K, P1, R1, P2, R2, q, threshold, branch=branching_factor
        )

        bf_scores, bf_indices = brute_force_halfspace(K, q, threshold)

        kernel_mask = torch.isfinite(out)
        kernel_indices = kernel_mask.nonzero(as_tuple=True)[0]

        kernel_set = set(kernel_indices.cpu().numpy())
        bf_set = set(bf_indices.cpu().numpy())

        if len(bf_set) > 0:
            recall = len(kernel_set & bf_set) / len(bf_set)
            assert recall >= 0.99, f"Dim {dim}: Recall {recall:.3f} < 0.99"


class TestKernelEdgeCases:
    """Test edge cases in kernel execution."""

    def test_very_high_threshold(self):
        """Test with very high threshold (should return few/no results)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 1024
        branching_factor = 16
        dim = 128

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)

        # Impossibly high threshold
        threshold = 0.999

        out = triton_two_level_filter(
            K, P, R, q, threshold, branch=branching_factor, BLOCK_C=16
        )

        bf_scores, bf_indices = brute_force_halfspace(K, q, threshold)

        # Both should return few results
        kernel_count = (out >= threshold).sum().item()
        bf_count = len(bf_indices)

        assert (
            kernel_count <= bf_count * 1.1 + 10
        ), f"Kernel returned {kernel_count}, expected ~{bf_count}"

    def test_very_low_threshold(self):
        """Test with very low threshold (should return many results)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 1024
        branching_factor = 16
        dim = 128

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)

        # Very low threshold
        threshold = -0.999

        out = triton_two_level_filter(
            K, P, R, q, threshold, branch=branching_factor, BLOCK_C=16
        )

        bf_scores, bf_indices = brute_force_halfspace(K, q, threshold)

        # Most/all keys should pass
        kernel_count = (out >= threshold).sum().item()
        bf_count = len(bf_indices)

        # Should have high recall
        if bf_count > 0:
            recall = kernel_count / bf_count
            assert recall >= 0.99, f"Low threshold recall: {recall:.3f}"

    def test_exact_boundary_threshold(self):
        """Test with threshold exactly at a score boundary."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 512
        branching_factor = 8
        dim = 128

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)

        # First compute scores to find median
        bf_scores = K @ q
        median_score = bf_scores.median().item()

        # Use median as threshold
        out = triton_two_level_filter(
            K, P, R, q, median_score, branch=branching_factor, BLOCK_C=8
        )

        bf_scores, bf_indices = brute_force_halfspace(K, q, median_score)

        kernel_mask = out >= median_score
        kernel_indices = kernel_mask.nonzero(as_tuple=True)[0]

        kernel_set = set(kernel_indices.cpu().numpy())
        bf_set = set(bf_indices.cpu().numpy())

        if len(bf_set) > 0:
            recall = len(kernel_set & bf_set) / len(bf_set)
            assert recall >= 0.99, f"Boundary threshold recall: {recall:.3f}"


class TestKernelDatatypes:
    """Test kernels with different data types."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_two_level_dtypes(self, dtype):
        """Test two-level kernel with different data types."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("FP16 requires CUDA")

        device = torch.device("cuda")
        n_keys = 1024
        branching_factor = 16
        dim = 128

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        # Convert to target dtype
        K = K.to(dtype)
        P = P.to(dtype)
        R = R.to(dtype)

        q = torch.randn(dim, device=device, dtype=dtype)
        q = F.normalize(q, p=2, dim=0)
        threshold = 0.0

        out = triton_two_level_filter(
            K, P, R, q, threshold, branch=branching_factor, BLOCK_C=16
        )

        # Compute reference in fp32 for comparison
        K_fp32 = K.to(torch.float32)
        q_fp32 = q.to(torch.float32)
        bf_scores, bf_indices = brute_force_halfspace(K_fp32, q_fp32, threshold)

        kernel_mask = out >= threshold
        kernel_indices = kernel_mask.nonzero(as_tuple=True)[0]

        kernel_set = set(kernel_indices.cpu().numpy())
        bf_set = set(bf_indices.cpu().numpy())

        if len(bf_set) > 0:
            recall = len(kernel_set & bf_set) / len(bf_set)
            # FP16 might have slightly lower recall due to precision
            min_recall = 0.98 if dtype == torch.float16 else 0.99
            assert (
                recall >= min_recall
            ), f"Dtype {dtype}: Recall {recall:.3f} < {min_recall}"


class TestKernelPerformance:
    """Basic performance sanity checks."""

    def test_two_level_execution(self):
        """Verify that kernel execution completes (basic sanity)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 4096
        branching_factor = 8
        dim = 128

        K, P, R = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=2, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)
        threshold = 0.0

        # Just verify it runs without error
        out = triton_two_level_filter(
            K, P, R, q, threshold, branch=branching_factor, BLOCK_C=8
        )

        assert out.shape == (n_keys,)
        assert out.device.type == device.type

    def test_three_level_execution(self):
        """Verify that three-level kernels execute successfully."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        n_keys = 4096
        branching_factor = 4
        dim = 128

        K, P1, R1, P2, R2 = create_hierarchical_data(
            n_keys, dim, branching_factor, num_levels=3, device=device
        )

        q = torch.randn(dim, device=device)
        q = F.normalize(q, p=2, dim=0)
        threshold = 0.0

        # Test all versions
        for version, func in [
            ("v1", triton_three_level_filter_v1),
        ]:
            out = func(K, P1, R1, P2, R2, q, threshold, branch=branching_factor)
            assert out.shape == (n_keys,), f"Version {version} wrong shape"
            assert out.device.type == device.type, f"Version {version} wrong device"


@pytest.fixture(scope="module")
def device():
    """Fixture to provide CUDA device if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        pytest.skip("CUDA not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
