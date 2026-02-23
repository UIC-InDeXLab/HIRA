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
        self._tmp_out: Optional[torch.Tensor] = None

    @staticmethod
    def _flatten_query(
        query: torch.Tensor, *, dim: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(query, torch.Tensor):
            raise TypeError(f"query must be a torch.Tensor, got {type(query)}")
        q = query.to(device=device, dtype=torch.float32).contiguous()
        if q.ndim == 4:
            if q.shape[0] != 1 or q.shape[2] != 1:
                raise ValueError(f"4D query must be (1,H,1,D), got {tuple(q.shape)}")
            q = q.squeeze(0).squeeze(-2)
        elif q.ndim != 2:
            raise ValueError(
                f"query must be (H,D) or (1,H,1,D), got {tuple(query.shape)}"
            )
        if q.shape[-1] != dim:
            raise ValueError(f"query dim mismatch: expected D={dim}, got {q.shape[-1]}")
        return q.contiguous()

    @staticmethod
    def _resolve_q_head_to_kv(
        *,
        num_query_heads: int,
        num_kv_heads: int,
        device: torch.device,
        q_head_to_kv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if q_head_to_kv is None:
            if num_query_heads == num_kv_heads:
                return torch.arange(num_query_heads, device=device, dtype=torch.long)
            if (num_query_heads % num_kv_heads) != 0:
                raise ValueError(
                    f"Unsupported head mapping: H_q={num_query_heads} must equal or be divisible by H_kv={num_kv_heads}"
                )
            group_size = num_query_heads // num_kv_heads
            return (
                torch.arange(num_query_heads, device=device, dtype=torch.long)
                // group_size
            )

        m = q_head_to_kv.to(device=device, dtype=torch.long).contiguous().reshape(-1)
        if m.numel() != num_query_heads:
            raise ValueError(
                f"q_head_to_kv length mismatch: expected H_q={num_query_heads}, got {m.numel()}"
            )
        if torch.any((m < 0) | (m >= num_kv_heads)):
            raise ValueError(f"q_head_to_kv values must be in [0, {num_kv_heads - 1}]")
        return m

    @staticmethod
    def _coerce_threshold(
        threshold: torch.Tensor, *, num_query_heads: int, device: torch.device
    ) -> torch.Tensor:
        t = threshold.to(device=device, dtype=torch.float32).contiguous().reshape(-1)
        if t.numel() == 1 and num_query_heads > 1:
            t = t.expand(num_query_heads)
        if t.numel() != num_query_heads:
            raise ValueError(
                f"threshold mismatch: expected scalar or H_q={num_query_heads}, got {t.numel()}"
            )
        return t.contiguous()

    def _prepare_output_buffer(
        self, *, num_query_heads: int, num_keys: int, indexer: CUDAIndexer
    ) -> torch.Tensor:
        num_kv_heads = int(indexer.children.shape[0])
        if (
            num_query_heads == num_kv_heads
            and indexer.buffer is not None
            and tuple(indexer.buffer.shape) == (num_kv_heads, num_keys)
        ):
            indexer.buffer.zero_()
            return indexer.buffer

        needs_new = (
            self._tmp_out is None
            or self._tmp_out.device != indexer.children.device
            or self._tmp_out.dtype != indexer.children.dtype
            or tuple(self._tmp_out.shape) != (num_query_heads, num_keys)
        )
        if needs_new:
            self._tmp_out = torch.zeros(
                (num_query_heads, num_keys),
                device=indexer.children.device,
                dtype=indexer.children.dtype,
            )
        else:
            self._tmp_out.zero_()
        return self._tmp_out

    def search(
        self,
        query,
        threshold,
        indexer: CUDAIndexer,
        q_head_to_kv: Optional[torch.Tensor] = None,
    ):
        # normalize query
        # query = query / torch.norm(query, p=2)
        # Query is already normalized
        if indexer.children is None:
            raise ValueError("Indexer is not built: missing children tensor.")
        dim = int(indexer.children.shape[2])
        num_kv_heads = int(indexer.children.shape[0])
        num_keys = int(indexer.children.shape[1])

        q = self._flatten_query(query=query, dim=dim, device=indexer.children.device)
        num_query_heads = int(q.shape[0])
        q_head_to_kv = self._resolve_q_head_to_kv(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            device=indexer.children.device,
            q_head_to_kv=q_head_to_kv,
        )
        t = self._coerce_threshold(
            threshold=threshold,
            num_query_heads=num_query_heads,
            device=indexer.children.device,
        )
        out = self._prepare_output_buffer(
            num_query_heads=num_query_heads, num_keys=num_keys, indexer=indexer
        )

        depth_value = getattr(indexer.depth, "value", indexer.depth)

        if depth_value == CUDAIndexer.DEPTH.TWO_LEVELS.value:
            output = triton_two_level_filter(
                indexer.children,
                indexer.parents,
                indexer.parent_radii,
                q,
                t,
                q_head_to_kv=q_head_to_kv,
                out=out,
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
                q,
                t,
                q_head_to_kv=q_head_to_kv,
                out=out,
                branch=indexer.branching_factor,
                BLOCK_C=self.block_c,
            )
        else:
            raise ValueError(f"Unsupported index depth: {indexer.depth}")
        return output

    def synthetic_scanned_fraction(
        self,
        query,
        threshold,
        indexer: CUDAIndexer,
        q_head_to_kv: Optional[torch.Tensor] = None,
    ):
        """
        Synthetic estimate of how many child rows are scanned by the CUDA traversal.

        Returns a dict with:
        - scanned_fraction_per_head: (H_q,) tensor in [0, 1]
        - scanned_children_per_head: (H_q,) tensor (counts)
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
        num_kv_heads = int(indexer.children.shape[0])
        dim = int(indexer.children.shape[2])

        q = self._flatten_query(query=query, dim=dim, device=device)
        num_query_heads = int(q.shape[0])
        q_head_to_kv = self._resolve_q_head_to_kv(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            device=device,
            q_head_to_kv=q_head_to_kv,
        )
        t = self._coerce_threshold(
            threshold=threshold,
            num_query_heads=num_query_heads,
            device=device,
        )

        depth_value = getattr(indexer.depth, "value", indexer.depth)
        bf = int(indexer.branching_factor)

        parents = indexer.parents.index_select(0, q_head_to_kv)
        parent_radii = indexer.parent_radii.index_select(0, q_head_to_kv)

        if depth_value == CUDAIndexer.DEPTH.TWO_LEVELS.value:
            # Parent gate: (q·p + r) > t
            parent_scores = torch.einsum("hmd,hd->hm", parents, q)
            parent_pass = (parent_scores + parent_radii) > t.unsqueeze(-1)
            scanned_children_per_head = parent_pass.sum(dim=1).to(torch.int64) * bf
        elif depth_value == CUDAIndexer.DEPTH.THREE_LEVELS.value:
            if indexer.grand_parents is None or indexer.grand_parent_radii is None:
                raise ValueError(
                    "Three-level search requires grand_parents and grand_parent_radii."
                )
            grand_parents = indexer.grand_parents.index_select(0, q_head_to_kv)
            grand_parent_radii = indexer.grand_parent_radii.index_select(
                0, q_head_to_kv
            )
            # Grandparent gate (P2 -> P1 mask pass)
            gp_scores = torch.einsum("hgd,hd->hg", grand_parents, q)
            gp_pass = (gp_scores + grand_parent_radii) > t.unsqueeze(-1)

            # Expand gp mask to level-1 parents exactly as branch-grouped layout.
            gp_mask_on_p1 = (
                gp_pass.unsqueeze(-1).expand(-1, -1, bf).reshape(gp_pass.shape[0], -1)
            )

            # Parent gate (P1 -> K), masked by grandparent pass.
            p1_scores = torch.einsum("hmd,hd->hm", parents, q)
            p1_pass = gp_mask_on_p1 & ((p1_scores + parent_radii) > t.unsqueeze(-1))
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
