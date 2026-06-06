import json
import math
import numpy as np
import torch

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from scipy.ndimage import (
        distance_transform_edt,
        gaussian_filter,
        binary_closing,
        binary_opening,
    )
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ------------------------------------------------------------
# Common helpers
# ------------------------------------------------------------

def to_numpy_mask_batch(masks):
    if isinstance(masks, torch.Tensor):
        arr = masks.detach().cpu().float().numpy()
    else:
        arr = np.asarray(masks, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(f"Expected MASK shape [B,H,W], got {arr.shape}")

    return arr.astype(np.float32)


def to_numpy_image_batch(images):
    if isinstance(images, torch.Tensor):
        arr = images.detach().cpu().float().numpy()
    else:
        arr = np.asarray(images, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[None, :, :, :]

    if arr.ndim != 4:
        raise ValueError(f"Expected IMAGE shape [B,H,W,C], got {arr.shape}")

    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def to_mask_tensor(arr):
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32))


def to_image_tensor(arr):
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32))


def mask_stats(mask, threshold=0.5):
    binary = mask > float(threshold)
    area = int(binary.sum())
    h, w = mask.shape

    if area <= 0:
        return {
            "area": 0,
            "mean": float(mask.mean()),
            "bbox": None,
            "bbox_area": 0,
            "cx": None,
            "cy": None,
            "touches_edge": False,
        }

    ys, xs = np.where(binary)
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    cx = float(xs.mean())
    cy = float(ys.mean())
    touches_edge = x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h

    return {
        "area": area,
        "mean": float(mask.mean()),
        "bbox": [x1, y1, x2, y2],
        "bbox_area": int(bbox_area),
        "cx": cx,
        "cy": cy,
        "touches_edge": bool(touches_edge),
    }


def expand_bbox(bbox, w, h, padding_px=0, square=False, min_size=8):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 -= padding_px
    y1 -= padding_px
    x2 += padding_px
    y2 += padding_px

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    if square:
        side = max(bw, bh, float(min_size))
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        x1 = cx - side * 0.5
        x2 = cx + side * 0.5
        y1 = cy - side * 0.5
        y2 = cy + side * 0.5
    else:
        if bw < min_size:
            cx = (x1 + x2) * 0.5
            x1 = cx - min_size * 0.5
            x2 = cx + min_size * 0.5
        if bh < min_size:
            cy = (y1 + y2) * 0.5
            y1 = cy - min_size * 0.5
            y2 = cy + min_size * 0.5

    # Clip while preserving reasonable crop size.
    x1 = int(round(max(0, min(w - 1, x1))))
    y1 = int(round(max(0, min(h - 1, y1))))
    x2 = int(round(max(x1 + 1, min(w, x2))))
    y2 = int(round(max(y1 + 1, min(h, y2))))

    return [x1, y1, x2, y2]


def resize_image_np(img, size):
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required for resizing in Mask Crop Stabilizer.")
    return cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def resize_mask_np(mask, size):
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required for resizing in Mask Crop Stabilizer.")
    return cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------
# Node 1: Mask Quality Filter
# ------------------------------------------------------------

class MaskQualityFilter:
    """
    Detects bad SAM/SAM3/SAM3.1 masks and zeros them so Mask Interpolator Pro can repair them.
    It catches more than just empty masks:
    - too small
    - too large
    - bbox too small
    - sudden centroid jumps
    - sudden area jumps
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "area_threshold": ("INT", {"default": 64, "min": 1, "max": 1000000}),
                "mean_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "max_area_fraction": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_bbox_area": ("INT", {"default": 64, "min": 1, "max": 1000000}),
                "max_centroid_jump_px": ("FLOAT", {"default": 160.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
                "max_area_jump_ratio": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "mode": (["zero_invalid", "keep_original"], {"default": "zero_invalid"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("filtered_masks", "invalid_frame_mask", "report")
    FUNCTION = "run"
    CATEGORY = "mask/video"

    def run(
        self,
        masks,
        area_threshold,
        mean_threshold,
        max_area_fraction,
        min_bbox_area,
        max_centroid_jump_px,
        max_area_jump_ratio,
        mode,
        threshold,
    ):
        arr = to_numpy_mask_batch(masks)
        B, H, W = arr.shape
        total_px = H * W

        stats = [mask_stats(arr[i], threshold=threshold) for i in range(B)]
        valid = [True] * B
        reasons = [[] for _ in range(B)]

        # Basic checks
        for i, s in enumerate(stats):
            if s["area"] < int(area_threshold):
                valid[i] = False
                reasons[i].append("area_small")
            if s["mean"] < float(mean_threshold):
                valid[i] = False
                reasons[i].append("mean_low")
            if s["area"] > float(max_area_fraction) * total_px:
                valid[i] = False
                reasons[i].append("area_too_large")
            if s["bbox"] is None or s["bbox_area"] < int(min_bbox_area):
                valid[i] = False
                reasons[i].append("bbox_small")

        # Temporal checks against last accepted valid mask.
        last_good = None
        for i in range(B):
            if not valid[i]:
                continue

            s = stats[i]
            if last_good is not None:
                p = stats[last_good]

                if p["cx"] is not None and s["cx"] is not None and float(max_centroid_jump_px) > 0:
                    dist = math.sqrt((s["cx"] - p["cx"]) ** 2 + (s["cy"] - p["cy"]) ** 2)
                    if dist > float(max_centroid_jump_px):
                        valid[i] = False
                        reasons[i].append(f"centroid_jump_{dist:.1f}px")

                if p["area"] > 0 and s["area"] > 0 and float(max_area_jump_ratio) > 1:
                    ratio = max(s["area"] / p["area"], p["area"] / s["area"])
                    if ratio > float(max_area_jump_ratio):
                        valid[i] = False
                        reasons[i].append(f"area_jump_{ratio:.2f}x")

            if valid[i]:
                last_good = i

        out = arr.copy()
        invalid_frame_mask = np.zeros_like(arr, dtype=np.float32)

        invalid_count = 0
        reason_counter = {}

        for i in range(B):
            if not valid[i]:
                invalid_count += 1
                invalid_frame_mask[i, :, :] = 1.0
                if mode == "zero_invalid":
                    out[i, :, :] = 0.0

                for r in reasons[i]:
                    key = r.split("_jump_")[0] if "_jump_" in r else r
                    reason_counter[key] = reason_counter.get(key, 0) + 1

        report_obj = {
            "node": "Mask Quality Filter",
            "frames": B,
            "valid": int(sum(valid)),
            "invalid": int(invalid_count),
            "mode": mode,
            "reasons": reason_counter,
            "note": "Use filtered_masks -> Mask Interpolator Pro. The invalid_frame_mask is white on frames marked invalid.",
        }

        return (to_mask_tensor(out), to_mask_tensor(invalid_frame_mask), json.dumps(report_obj, indent=2))


# ------------------------------------------------------------
# Node 2: Mask Interpolator Pro
# ------------------------------------------------------------

def postprocess_mask(mask, blur_sigma=0.8, threshold=0.5, morph_radius=1):
    out = mask.astype(np.float32)

    if HAS_SCIPY and blur_sigma > 0:
        out = gaussian_filter(out, sigma=float(blur_sigma))

    out = np.clip(out, 0.0, 1.0)

    if HAS_SCIPY and morph_radius > 0:
        binmask = out > float(threshold)
        binmask = binary_closing(binmask, iterations=int(morph_radius))
        binmask = binary_opening(binmask, iterations=max(1, int(morph_radius) // 2))
        out = binmask.astype(np.float32)

    return np.clip(out, 0.0, 1.0)


def is_valid_mask(mask, area_threshold=64, mean_threshold=0.001):
    area = int((mask > 0.5).sum())
    meanv = float(mask.mean())
    return area >= int(area_threshold) and meanv >= float(mean_threshold)


def signed_distance(mask, threshold=0.5):
    if not HAS_SCIPY:
        return mask.astype(np.float32) * 2.0 - 1.0

    inside = mask > float(threshold)
    outside = ~inside
    dist_in = distance_transform_edt(inside)
    dist_out = distance_transform_edt(outside)
    return (dist_in - dist_out).astype(np.float32)


def sdf_to_mask(sdf, sharpness=1.0):
    sharpness = max(1e-6, float(sharpness))
    soft = 1.0 / (1.0 + np.exp(-sdf / sharpness))
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def interpolate_sdf(mask_a, mask_b, t, sharpness=1.0):
    sdf_a = signed_distance(mask_a)
    sdf_b = signed_distance(mask_b)
    sdf = (1.0 - float(t)) * sdf_a + float(t) * sdf_b
    return sdf_to_mask(sdf, sharpness=sharpness)


def resize_for_flow(img, max_side=512):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale == 1.0:
        return img, 1.0
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def compute_flow(img1, img2):
    if not HAS_CV2:
        return None

    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)

    if img1.ndim == 3:
        g1 = cv2.cvtColor((img1 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        g1 = (img1 * 255.0).astype(np.uint8)

    if img2.ndim == 3:
        g2 = cv2.cvtColor((img2 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        g2 = (img2 * 255.0).astype(np.uint8)

    return cv2.calcOpticalFlowFarneback(
        g1,
        g2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )


def warp_mask(mask, flow):
    h, w = mask.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)

    return np.clip(
        cv2.remap(
            mask.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
        0.0,
        1.0,
    )


def interpolate_flow_guided(mask_a, img_a, mask_b, img_b, img_t, t):
    if not HAS_CV2:
        return interpolate_sdf(mask_a, mask_b, t)

    img_a_small, _ = resize_for_flow(img_a)
    img_b_small, _ = resize_for_flow(img_b)
    img_t_small, _ = resize_for_flow(img_t)

    target_h, target_w = img_t_small.shape[:2]

    def resize_img_to(img, shape_hw):
        return cv2.resize(img, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)

    img_a_small = resize_img_to(img_a_small, (target_h, target_w))
    img_b_small = resize_img_to(img_b_small, (target_h, target_w))

    mask_a_small = cv2.resize(mask_a.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask_b_small = cv2.resize(mask_b.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    flow_a_t = compute_flow(img_a_small, img_t_small)
    flow_b_t = compute_flow(img_b_small, img_t_small)

    if flow_a_t is None or flow_b_t is None:
        return interpolate_sdf(mask_a, mask_b, t)

    warp_a = warp_mask(mask_a_small, flow_a_t)
    warp_b = warp_mask(mask_b_small, flow_b_t)

    soft = (1.0 - float(t)) * warp_a + float(t) * warp_b
    soft = cv2.resize(soft, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_LINEAR)

    sdf_soft = interpolate_sdf(mask_a, mask_b, t, sharpness=1.0)
    out = 0.65 * soft + 0.35 * sdf_soft
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class MaskInterpolatorPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "area_threshold": ("INT", {"default": 64, "min": 1, "max": 1000000}),
                "mean_threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "method": (["auto", "flow_guided", "sdf"], {"default": "auto"}),
                "post_blur_sigma": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "morph_radius": ("INT", {"default": 1, "min": 0, "max": 10}),
                "max_gap": ("INT", {"default": 24, "min": 1, "max": 1000}),
            },
            "optional": {
                "images": ("IMAGE",),
                "invalid_frame_mask": ("MASK",),
                "beats_used": ("STRING", {"default": "", "multiline": True, "placeholder": "beats_used from FrameSequenceGenerator — window boundaries for interpolation"}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "report")
    FUNCTION = "run"
    CATEGORY = "mask/video"

    def run(
        self,
        masks,
        area_threshold,
        mean_threshold,
        method,
        post_blur_sigma,
        threshold,
        morph_radius,
        max_gap,
        images=None,
        invalid_frame_mask=None,
        beats_used="",
    ):
        masks_np = to_numpy_mask_batch(masks)
        images_np = to_numpy_image_batch(images) if images is not None else None

        B, H, W = masks_np.shape

        if images_np is not None and images_np.shape[0] != B:
            raise ValueError(f"IMAGE batch must have same batch size as MASK batch. images={images_np.shape[0]}, masks={B}")

        if invalid_frame_mask is not None:
            invalid_np = to_numpy_mask_batch(invalid_frame_mask)
            if invalid_np.shape[0] != B:
                raise ValueError(
                    f"invalid_frame_mask batch size must match masks. "
                    f"invalid={invalid_np.shape[0]}, masks={B}"
                )
            # A frame is valid unless its invalid_frame_mask has any non-zero pixel
            valid = [(invalid_np[i].max() < 0.5) for i in range(B)]
            valid_source = "invalid_frame_mask"
        else:
            valid = [
                is_valid_mask(masks_np[i], area_threshold=area_threshold, mean_threshold=mean_threshold)
                for i in range(B)
            ]
            valid_source = "internal"

        out = masks_np.copy()
        valid_idx = [i for i, v in enumerate(valid) if v]

        # --- Parse window boundaries from beats_used ---
        window_boundaries = set()
        try:
            bu = json.loads(beats_used or "[]")
            if isinstance(bu, list):
                for entry in bu:
                    offset = int(entry.get("batch_offset", -1))
                    count = int(entry.get("batch_frame_count", 0))
                    if offset >= 0 and count > 0:
                        last_frame = offset + count - 1
                        if last_frame < B - 1:
                            window_boundaries.add(last_frame)
        except Exception:
            pass

        if len(valid_idx) == 0:
            report = "Mask Interpolator Pro: No valid masks found; returned original masks unchanged."
            return (to_mask_tensor(out), report)

        first_valid = valid_idx[0]
        for i in range(0, first_valid):
            out[i] = out[first_valid]

        last_valid = valid_idx[-1]
        for i in range(last_valid + 1, B):
            out[i] = out[last_valid]

        repaired_count = 0
        large_gaps_fallback = 0
        boundaries_crossed = 0

        for k in range(len(valid_idx) - 1):
            left = valid_idx[k]
            right = valid_idx[k + 1]
            gap = right - left - 1

            if gap <= 0:
                continue

            # --- Window boundary check: don't interpolate across windows ---
            if window_boundaries:
                crossing = sorted([b for b in window_boundaries if left < b < right])
                if crossing:
                    boundary = crossing[0]
                    # Left side of boundary: clamp to left
                    for j in range(left + 1, boundary + 1):
                        out[j] = out[left]
                    # Right side of boundary: clamp to right
                    for j in range(boundary + 1, right):
                        out[j] = out[right]
                    repaired_count += (right - left - 1)
                    boundaries_crossed += 1
                    continue

            if gap > int(max_gap):
                for j in range(left + 1, right):
                    alpha = (j - left) / (right - left)
                    out[j] = out[left] if alpha < 0.5 else out[right]
                    repaired_count += 1
                large_gaps_fallback += 1
                continue

            for j in range(1, gap + 1):
                t = j / (gap + 1)
                target_idx = left + j

                use_flow = (
                    method in ["auto", "flow_guided"]
                    and images_np is not None
                    and HAS_CV2
                )

                if use_flow:
                    try:
                        out[target_idx] = interpolate_flow_guided(
                            out[left],
                            images_np[left],
                            out[right],
                            images_np[right],
                            images_np[target_idx],
                            t,
                        )
                    except Exception:
                        out[target_idx] = interpolate_sdf(out[left], out[right], t)
                else:
                    out[target_idx] = interpolate_sdf(out[left], out[right], t)

                repaired_count += 1

        for i in range(B):
            out[i] = postprocess_mask(
                out[i],
                blur_sigma=post_blur_sigma,
                threshold=threshold,
                morph_radius=morph_radius,
            )

        method_used = method
        if method == "auto":
            method_used = "flow_guided" if (images_np is not None and HAS_CV2) else "sdf"

        report_obj = {
            "node": "Mask Interpolator Pro",
            "frames": B,
            "valid_before": int(sum(valid)),
            "repaired": int(repaired_count),
            "large_gaps_fallback": int(large_gaps_fallback),
            "boundaries_crossed": int(boundaries_crossed),
            "method": method_used,
            "valid_source": valid_source,
            "window_boundaries": len(window_boundaries),
            "cv2": HAS_CV2,
            "scipy": HAS_SCIPY,
        }

        return (to_mask_tensor(out), json.dumps(report_obj, indent=2))


# ------------------------------------------------------------
# Node 3: Mask Crop Stabilizer
# ------------------------------------------------------------

def fill_bboxes_over_time(raw_bboxes, W, H):
    B = len(raw_bboxes)
    valid_idx = [i for i, b in enumerate(raw_bboxes) if b is not None]

    if not valid_idx:
        return [[0, 0, W, H] for _ in range(B)]

    xs = np.arange(B, dtype=np.float32)
    filled = np.zeros((B, 4), dtype=np.float32)

    for c in range(4):
        values = np.array([raw_bboxes[i][c] for i in valid_idx], dtype=np.float32)
        filled[:, c] = np.interp(xs, np.array(valid_idx, dtype=np.float32), values)

    return filled.tolist()


def smooth_bboxes(bboxes, smoothing=0.65):
    arr = np.asarray(bboxes, dtype=np.float32)
    B = arr.shape[0]

    if B <= 1 or smoothing <= 0:
        return arr

    s = float(np.clip(smoothing, 0.0, 0.98))

    # Forward EMA
    fwd = arr.copy()
    for i in range(1, B):
        fwd[i] = s * fwd[i - 1] + (1.0 - s) * arr[i]

    # Backward EMA
    bwd = arr.copy()
    for i in range(B - 2, -1, -1):
        bwd[i] = s * bwd[i + 1] + (1.0 - s) * arr[i]

    return (fwd + bwd) * 0.5


class MaskCropStabilizer:
    """
    Creates stable image crops from video frames + masks.
    This is important before DINOv2/SigLIP/CLIP embeddings, because unstable crops
    can create false visual jumps even when the outfit did not change.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "output_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "padding_percent": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 0.98, "step": 0.01}),
                "square_crop": ("BOOLEAN", {"default": True}),
                "min_crop_size": ("INT", {"default": 96, "min": 8, "max": 4096}),
                "mask_mode": (["image_crop", "masked_black", "masked_white"], {"default": "image_crop"}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "valid_area_threshold": ("INT", {"default": 64, "min": 1, "max": 1000000}),
            },
            "optional": {
                "beats_used": ("STRING", {"default": "", "multiline": True, "placeholder": "beats_used from FrameSequenceGenerator — reset smoothing at window boundaries"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("crops", "crop_masks", "report")
    FUNCTION = "run"
    CATEGORY = "mask/video"

    def run(
        self,
        images,
        masks,
        output_size,
        padding_percent,
        smoothing,
        square_crop,
        min_crop_size,
        mask_mode,
        mask_threshold,
        valid_area_threshold,
        beats_used="",
    ):
        if not HAS_CV2:
            raise RuntimeError("Mask Crop Stabilizer requires opencv-python. Install it with: pip install opencv-python")

        img_np = to_numpy_image_batch(images)
        mask_np = to_numpy_mask_batch(masks)

        B, H, W, C = img_np.shape

        if mask_np.shape[0] != B:
            raise ValueError(f"images and masks batch size must match. images={B}, masks={mask_np.shape[0]}")
        if mask_np.shape[1] != H or mask_np.shape[2] != W:
            raise ValueError(f"mask resolution must match image resolution. image={H}x{W}, mask={mask_np.shape[1]}x{mask_np.shape[2]}")

        raw_bboxes = []
        valid_count = 0

        for i in range(B):
            s = mask_stats(mask_np[i], threshold=mask_threshold)
            if s["bbox"] is not None and s["area"] >= int(valid_area_threshold):
                raw_bboxes.append(s["bbox"])
                valid_count += 1
            else:
                raw_bboxes.append(None)

        filled = fill_bboxes_over_time(raw_bboxes, W, H)

        # --- Window-aware smoothing: reset EMA at window boundaries ---
        window_boundaries = []
        try:
            bu = json.loads(beats_used or "[]")
            if isinstance(bu, list):
                for entry in bu:
                    offset = int(entry.get("batch_offset", -1))
                    count = int(entry.get("batch_frame_count", 0))
                    if offset >= 0 and count > 0 and offset + count < B:
                        window_boundaries.append(offset + count)
        except Exception:
            pass

        if window_boundaries and len(window_boundaries) > 0:
            # Split into windows and smooth each independently
            starts = [0] + window_boundaries
            ends = window_boundaries + [B]
            smoothed_parts = []
            for ws, we in zip(starts, ends):
                if we > ws:
                    win = filled[ws:we]
                    if len(win) > 1:
                        smoothed_parts.append(smooth_bboxes(win, smoothing=smoothing))
                    else:
                        smoothed_parts.append(win)
            smoothed = np.concatenate(smoothed_parts, axis=0) if smoothed_parts else filled
        else:
            smoothed = smooth_bboxes(filled, smoothing=smoothing)

        crops = np.zeros((B, int(output_size), int(output_size), C), dtype=np.float32)
        crop_masks = np.zeros((B, int(output_size), int(output_size)), dtype=np.float32)
        final_bboxes = []

        for i in range(B):
            x1, y1, x2, y2 = smoothed[i]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pad_px = int(round(max(bw, bh) * float(padding_percent)))

            bbox = expand_bbox(
                [x1, y1, x2, y2],
                W,
                H,
                padding_px=pad_px,
                square=bool(square_crop),
                min_size=int(min_crop_size),
            )
            final_bboxes.append(bbox)

            bx1, by1, bx2, by2 = bbox
            img_crop = img_np[i, by1:by2, bx1:bx2, :]
            m_crop = mask_np[i, by1:by2, bx1:bx2]

            if mask_mode != "image_crop":
                m3 = np.expand_dims(np.clip(m_crop, 0.0, 1.0), axis=-1)
                if mask_mode == "masked_black":
                    img_crop = img_crop * m3
                elif mask_mode == "masked_white":
                    img_crop = img_crop * m3 + (1.0 - m3)

            crops[i] = resize_image_np(img_crop, int(output_size))
            crop_masks[i] = resize_mask_np(m_crop, int(output_size))

        report_obj = {
            "node": "Mask Crop Stabilizer",
            "frames": B,
            "valid_bboxes": int(valid_count),
            "output_size": int(output_size),
            "smoothing": float(smoothing),
            "square_crop": bool(square_crop),
            "mask_mode": mask_mode,
            "first_bbox": final_bboxes[0] if final_bboxes else None,
            "last_bbox": final_bboxes[-1] if final_bboxes else None,
            "note": "Use crops as stable input for DINOv2/SigLIP/CLIP embeddings.",
        }

        return (to_image_tensor(crops), to_mask_tensor(crop_masks), json.dumps(report_obj, indent=2))


# ── Beat ↔ Change Synchronizer ───────────────────────────────────────

class BeatChangeSynchronizer:
    """Aligns audio beatdrops with DINOv2-detected outfit change points.

    Takes beats_used from FrameSequenceGenerator and change_frames from
    DINOv2FrameChangeDetector, then snaps each beatdrop to the nearest
    visual outfit change frame within max_distance.

    This ensures the beatdrop timing is synchronized with where the outfit
    ACTUALLY changes in the video, not just where the audio beat falls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "beats_used": ("STRING", {"default": "[]", "multiline": True,
                    "tooltip": "beats_used JSON from FrameSequenceGenerator"}),
                "change_frames": ("STRING", {"default": "[]", "multiline": True,
                    "tooltip": "change_frames_json from DINOv2FrameChangeDetector"}),
                "max_distance": ("INT", {"default": 30, "min": 1, "max": 300, "step": 1,
                    "tooltip": "Max frames a beatdrop can be snapped to reach a change point"}),
                "mode": (["snap_nearest", "snap_before", "snap_after"], {"default": "snap_nearest",
                    "tooltip": "snap_nearest: closest change frame. snap_before: only earlier. snap_after: only later."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("beats_used", "sync_report")
    FUNCTION = "sync"
    CATEGORY = "Amin/Researcher"

    def sync(self, beats_used, change_frames, max_distance, mode):
        try:
            beats = json.loads(beats_used or "[]")
        except json.JSONDecodeError:
            return (beats_used, json.dumps({"error": "Invalid beats_used JSON"}))

        try:
            changes = json.loads(change_frames or "[]")
        except json.JSONDecodeError:
            return (beats_used, json.dumps({"error": "Invalid change_frames JSON"}))

        if not beats or not changes:
            report = {"synced": 0, "total_beats": len(beats), "total_changes": len(changes),
                      "note": "No beats or changes to sync — returning original beats"}
            return (json.dumps(beats, indent=2), json.dumps(report, indent=2))

        # Extract change frame indices
        change_indices = [ch["frame_index"] for ch in changes if ch.get("is_outfit_change", True)]

        if not change_indices:
            report = {"synced": 0, "total_beats": len(beats), "total_changes": 0,
                      "note": "No outfit changes detected — returning original beats"}
            return (json.dumps(beats, indent=2), json.dumps(report, indent=2))

        synced_beats = []
        sync_log = []
        used_changes = set()

        for beat in beats:
            beat_copy = dict(beat)
            beat_frame = beat.get("frame_index", beat.get("batch_offset", 0))
            best_dist = max_distance + 1
            best_change = None

            for ci in change_indices:
                dist = abs(beat_frame - ci)
                if dist > max_distance:
                    continue

                # Filter by mode
                if mode == "snap_before" and ci > beat_frame:
                    continue
                if mode == "snap_after" and ci < beat_frame:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_change = ci

            if best_change is not None:
                beat_copy["frame_index"] = best_change
                beat_copy["original_frame_index"] = beat_frame
                beat_copy["synced_to_change"] = True
                beat_copy["sync_distance"] = best_dist
                used_changes.add(best_change)
                sync_log.append({
                    "beat_index": beat.get("beat_index"),
                    "original_frame": beat_frame,
                    "synced_frame": best_change,
                    "distance": best_dist,
                })

            synced_beats.append(beat_copy)

        report = {
            "synced": len(sync_log),
            "total_beats": len(beats),
            "total_changes": len(change_indices),
            "changes_used": len(used_changes),
            "max_distance": int(max_distance),
            "mode": mode,
            "sync_details": sync_log,
            "unmatched_changes": [ci for ci in change_indices if ci not in used_changes],
        }

        return (
            json.dumps(synced_beats, indent=2),
            json.dumps(report, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "MaskQualityFilter": MaskQualityFilter,
    "MaskInterpolatorPro": MaskInterpolatorPro,
    "MaskCropStabilizer": MaskCropStabilizer,
    "BeatChangeSynchronizer": BeatChangeSynchronizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskQualityFilter": "Mask Quality Filter",
    "MaskInterpolatorPro": "Mask Interpolator Pro",
    "MaskCropStabilizer": "Mask Crop Stabilizer",
    "BeatChangeSynchronizer": "Beat ↔ Change Synchronizer",
}
