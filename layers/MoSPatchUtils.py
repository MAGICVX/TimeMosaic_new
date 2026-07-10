"""
MoS Patch Utilities — hierarchical splitting helpers.

Adapted from Kairos (refer/Kairos/tsfm/model/kairos/patch_utils.py).
These functions implement the core MoS hierarchical splitting logic:
  1. _divide_patches: halve each patch that should be split
  2. _update_parent_mapping / _update_position_mapping: track split ancestry
  3. _create_granularity_mask: per-granularity activation masks
  4. _generate_x_final: rearrange features for MultiInSizeLinear
"""

from typing import Optional, Tuple
import torch


# ─── Core splitting ───────────────────────────────────────────────────────

def _divide_patches(
    x: torch.Tensor,
    size: torch.Tensor,
    to_divide: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    expert_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Halve each region that should be split into two child patches.

    Args:
        x:              [batch, patch_num, max_patch_size]  region windows
        size:           [batch, patch_num]                  effective size per region
        to_divide:      [batch, patch_num] bool             which regions to split
        weights:        [batch, patch_num, *] optional      routing weights to propagate
        expert_indices: [batch, patch_num, *] optional      expert assignments to propagate

    Returns:
        new_x, new_size, new_weights, new_expert_indices
        Shapes: new_patch_num = patch_num + max_div_count across batch
    """
    batch, patch_len, patch_size = x.shape

    div_counts = to_divide.sum(dim=1)
    if div_counts.max().item() == 0:
        if weights is not None:
            return x, size, weights, expert_indices
        return x, size, weights, expert_indices

    new_patch_len: int = patch_len + div_counts.max().item()

    batch_idx = torch.arange(batch, device=x.device)[:, None].expand(-1, patch_len)
    base_idx = torch.arange(patch_len, device=x.device)[None, :].expand(batch, -1)
    offset = torch.cumsum(to_divide.float(), dim=1).long()
    new_positions = base_idx + offset

    new_x = torch.zeros(batch, new_patch_len, patch_size, device=x.device, dtype=x.dtype)
    new_size = torch.zeros(batch, new_patch_len, device=size.device, dtype=size.dtype)

    new_weights = None
    if weights is not None:
        new_weights_shape = (batch, new_patch_len) + weights.shape[2:]
        new_weights = torch.zeros(new_weights_shape, dtype=weights.dtype, device=weights.device)

    new_expert_indices = None
    if expert_indices is not None:
        new_expert_indices_shape = (batch, new_patch_len) + expert_indices.shape[2:]
        new_expert_indices = torch.zeros(new_expert_indices_shape,
                                         dtype=expert_indices.dtype, device=expert_indices.device)

    # Scatter undivided patches
    undivided = ~to_divide
    new_x[batch_idx[undivided], new_positions[undivided]] = x[undivided]
    new_size[batch_idx[undivided], new_positions[undivided]] = size[undivided]
    if weights is not None:
        new_weights[batch_idx[undivided], new_positions[undivided]] = weights[undivided]
    if expert_indices is not None:
        new_expert_indices[batch_idx[undivided], new_positions[undivided]] = expert_indices[undivided]

    # Scatter divided patches into two halves
    divided = to_divide
    div_sizes = size[divided].div(2, rounding_mode="floor")

    # First half
    first_half_idx = torch.arange(patch_size, device=x.device)[None, :] < div_sizes[:, None]
    new_x[batch_idx[divided], new_positions[divided] - 1] = torch.where(
        first_half_idx, x[divided], torch.zeros_like(x[divided]))
    new_size[batch_idx[divided], new_positions[divided] - 1] = div_sizes
    if weights is not None:
        new_weights[batch_idx[divided], new_positions[divided] - 1] = weights[divided]
    if expert_indices is not None:
        new_expert_indices[batch_idx[divided], new_positions[divided] - 1] = expert_indices[divided]

    # Second half
    second_half_idx = (torch.arange(patch_size, device=x.device)[None, :] >= div_sizes[:, None]) & (
        torch.arange(patch_size, device=x.device)[None, :] < size[divided][:, None])
    second_half_values = torch.where(second_half_idx, x[divided], torch.zeros_like(x[divided]))
    second_half_values_rolled = torch.roll(second_half_values, -div_sizes.max().item(), dims=1)
    new_x[batch_idx[divided], new_positions[divided]] = second_half_values_rolled
    new_size[batch_idx[divided], new_positions[divided]] = size[divided] - div_sizes
    if weights is not None:
        new_weights[batch_idx[divided], new_positions[divided]] = weights[divided]
    if expert_indices is not None:
        new_expert_indices[batch_idx[divided], new_positions[divided]] = expert_indices[divided]

    return new_x, new_size, new_weights, new_expert_indices


# ─── Parent mapping tracking ──────────────────────────────────────────────

def _update_parent_mapping(
    parent_mapping: torch.Tensor,
    to_divide: torch.Tensor,
    div_counts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """After splitting, update which original patch each child belongs to."""
    batch, current_patch_num = parent_mapping.shape
    new_patch_nums = current_patch_num + div_counts
    max_new_patch_num = new_patch_nums.max()

    repeat_counts = 1 + to_divide.long()
    new_parent_mapping = torch.full((batch, max_new_patch_num), -1, dtype=torch.long, device=device)
    positions = torch.cumsum(repeat_counts, dim=1) - repeat_counts
    batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, current_patch_num)

    # First copy (always exists)
    valid_mask_1 = positions < max_new_patch_num
    new_parent_mapping[batch_idx[valid_mask_1], positions[valid_mask_1]] = parent_mapping[valid_mask_1]

    # Second copy (only for divided patches)
    second_positions = positions + 1
    valid_mask_2 = to_divide.bool() & (second_positions < max_new_patch_num)
    new_parent_mapping[batch_idx[valid_mask_2], second_positions[valid_mask_2]] = parent_mapping[valid_mask_2]

    return new_parent_mapping


def _update_position_mapping(
    position_mapping: torch.Tensor,
    to_divide: torch.Tensor,
    div_counts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """After splitting, update [start, end] positions within original patches."""
    batch, current_patch_num, _ = position_mapping.shape
    new_patch_nums = current_patch_num + div_counts
    max_new_patch_num = new_patch_nums.max()

    repeat_counts = 1 + to_divide.long()
    new_position_mapping = torch.zeros((batch, max_new_patch_num, 2), dtype=torch.long, device=device)
    positions = torch.cumsum(repeat_counts, dim=1) - repeat_counts
    batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, current_patch_num)

    # Undivided: keep original positions
    valid_mask_1 = positions < max_new_patch_num
    new_position_mapping[batch_idx[valid_mask_1], positions[valid_mask_1]] = position_mapping[valid_mask_1]

    # Divided: split the range
    second_positions = positions + 1
    valid_mask_2 = to_divide.bool() & (second_positions < max_new_patch_num)

    if valid_mask_2.any():
        divided_position_mapping = position_mapping[valid_mask_2]
        orig_start = divided_position_mapping[:, 0]
        orig_end = divided_position_mapping[:, 1]
        mid_pos = (orig_start + orig_end) // 2

        batch_indices_divided = batch_idx[valid_mask_2]
        first_positions_divided = positions[valid_mask_2]
        second_positions_divided = second_positions[valid_mask_2]

        new_position_mapping[batch_indices_divided, first_positions_divided, 0] = orig_start
        new_position_mapping[batch_indices_divided, first_positions_divided, 1] = mid_pos
        new_position_mapping[batch_indices_divided, second_positions_divided, 0] = mid_pos
        new_position_mapping[batch_indices_divided, second_positions_divided, 1] = orig_end

    return new_position_mapping


# ─── Granularity mask ─────────────────────────────────────────────────────

def _create_granularity_mask(
    original_expert_indices: torch.Tensor,
    parent_mapping: torch.Tensor,
    position_mapping: torch.Tensor,
    target_shape: Tuple[int, int, int],
    original_patch_len: int,
) -> torch.Tensor:
    """
    Create granularity mask for each child patch.

    Returns:
        granularity_mask: [batch, new_patch_num, max_granularity, original_patch_len]
    """
    batch, new_patch_num, patch_len = target_shape
    max_granularity = original_expert_indices.shape[-1]
    device = parent_mapping.device

    granularity_mask = torch.zeros(
        (batch, new_patch_num, max_granularity, original_patch_len),
        dtype=torch.float32, device=device)

    valid_mask = parent_mapping >= 0
    safe_parent_mapping = torch.clamp(parent_mapping, min=0)
    batch_indices = torch.arange(batch, device=device).view(-1, 1)

    patch_expert_indices = original_expert_indices[batch_indices, safe_parent_mapping]

    # Compute positions
    start_positions, end_positions = _compute_patch_positions(
        parent_mapping, new_patch_num, original_patch_len, batch, device)

    for granularity in range(max_granularity):
        has_granularity = (patch_expert_indices == granularity).any(dim=-1) & valid_mask
        if has_granularity.any():
            mask_pattern = _generate_granularity_pattern(
                granularity, start_positions, end_positions, original_patch_len,
                has_granularity, max_granularity, device)
            granularity_mask[:, :, granularity, :] = mask_pattern

    return granularity_mask


def _compute_patch_positions(
    parent_mapping: torch.Tensor,
    new_patch_num: int,
    original_patch_len: int,
    batch: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute start/end positions within original patches for each child."""
    start_positions = torch.zeros((batch, new_patch_num), dtype=torch.long, device=device)
    end_positions = torch.full((batch, new_patch_num), original_patch_len, dtype=torch.long, device=device)

    valid_mask = parent_mapping >= 0
    if not valid_mask.any():
        return start_positions, end_positions

    max_parent = parent_mapping.max().item() if parent_mapping.numel() > 0 else 0
    parent_values = torch.arange(max_parent + 1, device=device)
    parent_masks = (parent_mapping.unsqueeze(-1) == parent_values.unsqueeze(0).unsqueeze(0))

    patches_per_parent = parent_masks.sum(dim=1)
    cumsum_masks = torch.cumsum(parent_masks.float(), dim=1)
    relative_indices = (cumsum_masks - 1) * parent_masks.float()

    patch_sizes = original_patch_len // patches_per_parent.clamp(min=1)
    start_positions_expanded = (relative_indices * patch_sizes.unsqueeze(1)).long()
    end_positions_expanded = start_positions_expanded + patch_sizes.unsqueeze(1)

    final_start = start_positions_expanded.sum(dim=-1)
    final_end = end_positions_expanded.sum(dim=-1)

    start_positions = torch.where(valid_mask, final_start, start_positions)
    end_positions = torch.where(valid_mask, final_end.clamp(max=original_patch_len), end_positions)

    return start_positions, end_positions


def _generate_granularity_pattern(
    granularity: int,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
    original_patch_len: int,
    has_granularity: torch.Tensor,
    max_granularity: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate activation pattern for a specific granularity level."""
    batch, new_patch_num = has_granularity.shape
    mask_pattern = torch.zeros((batch, new_patch_num, original_patch_len), dtype=torch.float32, device=device)

    if not has_granularity.any():
        return mask_pattern

    if granularity == 0:
        mask_pattern[has_granularity] = 1.0
        return mask_pattern

    activation_length = max(1, original_patch_len // (2 ** granularity))
    batch_idx, patch_idx = torch.where(has_granularity)

    if len(batch_idx) > 0:
        start_pos = start_positions[batch_idx, patch_idx]
        slot_indices = start_pos // activation_length
        activation_starts = slot_indices * activation_length
        activation_ends = torch.clamp(activation_starts + activation_length, max=original_patch_len)

        pos_grid = torch.arange(original_patch_len, device=device).unsqueeze(0)
        range_masks = (pos_grid >= activation_starts.unsqueeze(1)) & (pos_grid < activation_ends.unsqueeze(1))
        mask_pattern[batch_idx, patch_idx] = range_masks.float()

    return mask_pattern


# ─── x_final generation ───────────────────────────────────────────────────

def _generate_x_final(
    parent_blocks: torch.Tensor,
    parent_blocks_mask: torch.Tensor,
    granularity_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange parent block features for MultiInSizeLinear consumption.

    Args:
        parent_blocks:      [batch, patch_num, patch_len]
        parent_blocks_mask: [batch, patch_num, patch_len]
        granularity_mask:   [batch, patch_num, max_granularity, patch_len]

    Returns:
        x_final: [max_granularity, batch * patch_num, max_feat_size * 2]
    """
    batch, patch_num, patch_len = parent_blocks.shape
    max_granularity = granularity_mask.shape[2]
    total_patches = batch * patch_num

    feat_sizes = [max(1, patch_len // (2 ** i)) for i in range(max_granularity)]
    max_feat_size = max(feat_sizes) if feat_sizes else patch_len

    x_final = torch.zeros(
        (max_granularity, total_patches, max_feat_size * 2),
        dtype=parent_blocks.dtype, device=parent_blocks.device)

    parent_blocks_flat = parent_blocks.reshape(total_patches, patch_len)
    parent_blocks_mask_flat = parent_blocks_mask.reshape(total_patches, patch_len)
    granularity_mask_flat = granularity_mask.reshape(total_patches, max_granularity, patch_len)

    for granularity in range(max_granularity):
        feat_size = feat_sizes[granularity]
        current_granularity_mask = granularity_mask_flat[:, granularity, :]

        masked_blocks = parent_blocks_flat * current_granularity_mask
        masked_mask = parent_blocks_mask_flat * current_granularity_mask

        rearranged_blocks = _rearrange_effective_values(
            masked_blocks, current_granularity_mask, feat_size, max_feat_size)
        rearranged_mask = _rearrange_effective_values(
            masked_mask, current_granularity_mask, feat_size, max_feat_size)

        rearranged_features = torch.cat([rearranged_blocks, rearranged_mask], dim=-1)
        x_final[granularity] = rearranged_features

    return x_final


def _rearrange_effective_values(
    features: torch.Tensor,
    mask: torch.Tensor,
    target_feat_size: int,
    max_feat_size: int,
) -> torch.Tensor:
    """Rearrange effective (masked) values to front positions."""
    total_patches, feat_dim = features.shape
    device = features.device

    rearranged = torch.zeros((total_patches, max_feat_size), dtype=features.dtype, device=device)
    effective_mask = mask > 0

    position_indices = torch.arange(feat_dim, device=device).unsqueeze(0).expand(total_patches, -1)
    masked_positions = torch.where(effective_mask, position_indices.float(), float('inf'))
    sorted_positions, sort_indices = torch.sort(masked_positions, dim=1)

    effective_counts = effective_mask.sum(dim=1)
    output_indices = torch.arange(target_feat_size, device=device).unsqueeze(0)
    take_counts = torch.clamp(effective_counts, max=target_feat_size).unsqueeze(1)
    valid_output_mask = output_indices < take_counts

    batch_indices, output_positions = torch.where(valid_output_mask)
    if len(batch_indices) > 0:
        source_indices = sort_indices[batch_indices, output_positions]
        source_values = features[batch_indices, source_indices]
        rearranged[batch_indices, output_positions] = source_values

    return rearranged
