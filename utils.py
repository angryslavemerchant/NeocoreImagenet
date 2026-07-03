import torch
import torch.nn.functional as F


def extract_patch(
    image: torch.Tensor,
    pos: torch.Tensor,
    patch_size: int,
    image_size: int,
) -> torch.Tensor:
    """
    Differentiably extract a square patch from a batch of images.

    Uses bilinear interpolation via F.grid_sample so gradients flow
    back through the position coordinates into MoveNet.

    Coordinate convention: normalized [-1, 1] with align_corners=True,
    meaning -1 is the center of the top-left pixel and 1 is the center
    of the bottom-right pixel. grid_sample expects (x, y) ordering.

    Padding: 'border' — out-of-bounds samples replicate the nearest edge
    pixel, producing a visible stripe artifact that the CNN can learn to
    associate with image boundaries.

    Args:
        image:      (B, C, H, W)
        pos:        (B, 2) patch center in normalized coords, (x, y) order
        patch_size: patch side length in pixels
        image_size: full image side length in pixels (assumed square)

    Returns:
        patch: (B, C, patch_size, patch_size)
    """
    B = image.size(0)

    # One pixel expressed in normalized coords (align_corners=True)
    pixel_size = 2.0 / (image_size - 1)

    # Offsets from center for each pixel position in the patch.
    # For patch_size=32: [-15.5, -14.5, ..., 14.5, 15.5] * pixel_size
    offsets = (
        torch.arange(patch_size, dtype=torch.float32, device=image.device)
        - (patch_size - 1) / 2.0
    ) * pixel_size  # (patch_size,)

    # Build 2D grid of (x, y) sample coordinates, centered at origin.
    # grid_sample expects last dim as (x, y).
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")  # (P, P)
    grid = torch.stack([grid_x, grid_y], dim=-1)                       # (P, P, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)                    # (B, P, P, 2)

    # Shift the centered grid to the requested patch position
    grid = grid + pos.view(B, 1, 1, 2)

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def clamp_pos(pos: torch.Tensor, patch_size: int, image_size: int) -> torch.Tensor:
    """
    Clamp patch center so the entire patch stays within image bounds.

    With align_corners=True, the valid range for the patch center is
    [-1 + half_patch, 1 - half_patch] where half_patch is the distance
    from center to the furthest sample point in normalized coords.

    Args:
        pos:        (B, 2) in normalized coords
        patch_size: patch side length in pixels
        image_size: full image side length in pixels

    Returns:
        clamped pos: (B, 2)
    """
    pixel_size = 2.0 / (image_size - 1)
    # Furthest offset from center in the patch
    half_patch = (patch_size - 1) / 2.0 * pixel_size
    return pos.clamp(-1.0 + half_patch, 1.0 - half_patch)


class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1, 5),
) -> list:
    """
    Compute top-k classification accuracy.

    Args:
        output: (B, num_classes) logits
        target: (B,) integer class indices
        topk:   tuple of k values to evaluate

    Returns:
        list of float accuracy percentages, one per k
    """
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                                        # (maxk, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # (maxk, B)

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append((correct_k / B * 100).item())
        return results
