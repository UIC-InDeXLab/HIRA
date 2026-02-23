from .base import BaseSearcher
from indexer import CUDAIndexer
import torch
from typing import Optional

from hira.kernels.triton_search_wrappers import (
    triton_two_level_filter,
    triton_three_level_filter_v1,
    triton_three_level_filter_kernel_v1,
    triton_three_level_filter_kernel_v2,
)


class CUDASearcher(BaseSearcher):

    def __init__(self, block_c):
        super().__init__()
        self.block_c = block_c

    def search(
        self,
        query,
        threshold,
        indexer: CUDAIndexer,
    ):
        # normalize query
        # query = query / torch.norm(query, p=2)
        # Query is already normalized

        indexer.buffer.zero_()

        depth_value = getattr(indexer.depth, "value", indexer.depth)

        if depth_value == CUDAIndexer.DEPTH.TWO_LEVELS.value:
            output = triton_two_level_filter(
                indexer.children,
                indexer.parents,
                indexer.parent_radii,
                query,
                threshold,
                out=indexer.buffer,  # check
                BLOCK_C=self.block_c,
                branch=indexer.branching_factor,
            )
        elif depth_value == CUDAIndexer.DEPTH.THREE_LEVELS.value:
            output = triton_three_level_filter_kernel_v1(
                indexer.children,
                indexer.parents,
                indexer.parent_radii,
                indexer.grand_parents,
                indexer.grand_parent_radii,
                query,
                threshold,
                out=indexer.buffer,
                branch=indexer.branching_factor,
                BLOCK_C=self.block_c,
            )

        else:
            raise ValueError(f"Unsupported index depth: {indexer.depth}")
        return output

    def synthetic_scanned_fraction(self, query, threshold, indexer: CUDAIndexer):
        """
        Synthetic estimate of how many child rows are scanned by the CUDA traversal.

        Returns a dict with:
        - scanned_fraction_per_head: (H,) tensor in [0, 1]
        - scanned_children_per_head: (H,) tensor (counts)
        - total_children: int
        - scanned_fraction_mean: float
        """
        if indexer.children is None:
            raise ValueError("Indexer is not built: missing children tensor.")
        if indexer.parents is None or indexer.parent_radii is None:
            raise ValueError(
                "Indexer is not built: missing parents/parent_radii tensors."
            )

        device = indexer.children.device
        num_heads = int(indexer.children.shape[0])
        dim = int(indexer.children.shape[2])

        # Accept common query layouts (H,D), (1,H,1,D), etc., then flatten to (H,D).
        q = query.to(device=device, dtype=torch.float32).contiguous().reshape(-1, dim)
        if q.shape[0] != num_heads:
            raise ValueError(
                f"query head mismatch after reshape: expected H={num_heads}, got {q.shape[0]} from shape={tuple(query.shape)}"
            )

        # Accept scalar or per-head threshold and flatten to (H,).
        t = threshold.to(device=device, dtype=torch.float32).contiguous().reshape(-1)
        if t.numel() == 1 and num_heads > 1:
            t = t.expand(num_heads)
        if t.numel() != num_heads:
            raise ValueError(
                f"threshold mismatch: expected scalar or H={num_heads}, got {t.numel()} from shape={tuple(threshold.shape)}"
            )

        depth_value = getattr(indexer.depth, "value", indexer.depth)
        bf = int(indexer.branching_factor)

        if depth_value == CUDAIndexer.DEPTH.TWO_LEVELS.value:
            # Parent gate: (q·p + r) > t
            parent_scores = torch.einsum("hmd,hd->hm", indexer.parents, q)
            parent_pass = (parent_scores + indexer.parent_radii) > t.unsqueeze(-1)
            scanned_children_per_head = parent_pass.sum(dim=1).to(torch.int64) * bf
        elif depth_value == CUDAIndexer.DEPTH.THREE_LEVELS.value:
            if indexer.grand_parents is None or indexer.grand_parent_radii is None:
                raise ValueError(
                    "Three-level search requires grand_parents and grand_parent_radii."
                )
            # Grandparent gate (P2 -> P1 mask pass)
            gp_scores = torch.einsum("hgd,hd->hg", indexer.grand_parents, q)
            gp_pass = (gp_scores + indexer.grand_parent_radii) > t.unsqueeze(-1)

            # Expand gp mask to level-1 parents exactly as branch-grouped layout.
            gp_mask_on_p1 = (
                gp_pass.unsqueeze(-1).expand(-1, -1, bf).reshape(gp_pass.shape[0], -1)
            )

            # Parent gate (P1 -> K), masked by grandparent pass.
            p1_scores = torch.einsum("hmd,hd->hm", indexer.parents, q)
            p1_pass = gp_mask_on_p1 & (
                (p1_scores + indexer.parent_radii) > t.unsqueeze(-1)
            )
            scanned_children_per_head = p1_pass.sum(dim=1).to(torch.int64) * bf
        else:
            raise ValueError(f"Unsupported index depth: {indexer.depth}")

        total_children = int(indexer.children.shape[1])
        denom = max(1, total_children)
        scanned_fraction_per_head = scanned_children_per_head.to(torch.float32) / float(
            denom
        )

        return {
            "scanned_fraction_per_head": scanned_fraction_per_head,
            "scanned_children_per_head": scanned_children_per_head,
            "total_children": total_children,
            "scanned_fraction_mean": float(scanned_fraction_per_head.mean().item()),
        }
