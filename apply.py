# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/


import os

# ---- GPU selection (must happen before importing TensorFlow) ----
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU 0 only. Change it if you have multiple GPUs and want to use another GPU for training.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF log spam
# Must be before TensorFlow is imported, otherwise it won’t take effect.

import argparse
import itertools
import numpy as np
import pathlib
import tifffile
import tensorflow as tf

import rcan.model   # ensures Lambda deserialization sees rcan.model
from rcan.utils import (
    apply,
    convert_to_multi_gpu_model,
    get_model_path,
    load_model,
    normalize,
    rescale,
    save_tiff)


def tuple_of_ints(string):
    return tuple(int(s) for s in string.split(','))


def percentile(x):
    x = float(x)
    if 0.0 <= x <= 100.0:
        return x
    else:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 100.0]")


def configure_tf():
    """Configure visible GPU + memory growth. Must be called before heavy TF usage."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("Visible GPUs:", gpus)
    else:
        print("WARNING: No GPU detected. Inference will run on CPU.")


def normalize_between_zero_and_one(m):
    max_val, min_val = m.max(), m.min()
    diff = max_val - min_val
    return (m - min_val) / diff if diff > 0 else np.zeros_like(m)


def save_restored_tiff(path: str, img: np.ndarray, fmt: str = "imagej"):
    """
    Save a 2D/3D restored image using tifffile with explicit axis handling.
    img should be:
      - 2D: (Y, X)
      - 3D: (Z, Y, X)
    """
    img = np.asarray(img)

    if img.ndim == 2:
        axes = "YX"
    elif img.ndim == 3:
        axes = "ZYX"
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

    if fmt == "imagej":
        # ImageJ interprets 3D arrays as Z-stacks when passed as (Z, Y, X)
        tifffile.imwrite(path, img, imagej=True, metadata={"axes": axes})
    elif fmt == "ome":
        # OME-TIFF: provide axes explicitly to avoid ambiguity
        tifffile.imwrite(path, img, ome=True, metadata={"axes": axes})
    else:
        raise ValueError(f"Unknown format: {fmt} (use 'imagej' or 'ome')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument(
        "-f", "--output_tiff_format", type=str,
        choices=["imagej", "ome"], default="imagej")
    parser.add_argument("-g", "--ground_truth", type=str)
    parser.add_argument("-b", "--bpp", type=int, choices=[8, 16, 32], default=32)
    parser.add_argument("-B", "--block_shape", type=tuple_of_ints)
    parser.add_argument("-O", "--block_overlap_shape", type=tuple_of_ints)
    parser.add_argument("--p_min", type=percentile, default=2.0)
    parser.add_argument("--p_max", type=percentile, default=99.9)
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--normalize_output_range_between_zero_and_one", action="store_true")
    args = parser.parse_args()

    if args.rescale and args.normalize_output_range_between_zero_and_one:
        raise ValueError(
            "You cannot set both `rescale` and `normalize_output_range_between_zero_and_one` at the same time")

    configure_tf()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    # Create output dir if needed
    if input_path.is_dir() and not output_path.exists():
        print("Creating output directory", output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    # Input/output path type must match (dir->dir or file->file)
    if input_path.is_dir() != output_path.is_dir():
        raise ValueError("Mismatch between input and output path types")

    # if args.ground_truth is None:
    #     gt_path = None
    # else:
    #     gt_path = pathlib.Path(args.ground_truth)
    #     if input_path.is_dir() != gt_path.is_dir():
    #         raise ValueError("Mismatch between input and ground truth path types")

    # Build file list
    if input_path.is_dir():
        raw_files = sorted(input_path.glob("*.tif"))
        if not raw_files:
            raise RuntimeError(f"No TIFF files found in {input_path}")

        # if gt_path is None:
        #     data = itertools.zip_longest(raw_files, [])
        # else:
        #     gt_files = sorted(gt_path.glob("*.tif"))
        #     if len(raw_files) != len(gt_files):
        #         raise ValueError(
        #             f"Mismatch between raw and ground truth file counts ({len(raw_files)} vs. {len(gt_files)})")
        #     data = zip(raw_files, gt_files)
        data = raw_files
    else:
        # data = [(input_path, gt_path)]
        data = [input_path]

    # Load model
    model_path = get_model_path(args.model_dir)
    print("Loading model from", model_path)

    # model = convert_to_multi_gpu_model(load_model(str(model_path), input_shape=args.block_shape))
    # Single-GPU: just load the model. (Do NOT call multi_gpu_model in TF2.)
    model = load_model(str(model_path), input_shape=args.block_shape)

    # Determine default overlap if not provided
    if args.block_overlap_shape is None:
        # overlap_shape = [max(1, x // 8) if x > 2 else 0 for x in model.input.shape.as_list()[1:-1]]

        in_shape = model.input.shape[1:-1]  # TensorShape
        # Convert to ints where possible; fall back safely if None
        in_shape = [int(d) if d is not None else 0 for d in in_shape]
        overlap_shape = [max(2, x // 8) if x > 2 else 0 for x in in_shape]
        overlap_shape = tuple(overlap_shape)
    else:
        overlap_shape = args.block_overlap_shape

    print("Using overlap_shape =", overlap_shape)

    # for raw_file, gt_file in data:
    for raw_file in data:
        print("Loading raw image from", raw_file)
        raw = tifffile.imread(str(raw_file))
        raw = normalize(raw, args.p_min, args.p_max)

        print("Applying model")
        restored = apply(model, raw, overlap_shape=overlap_shape, verbose=True)
        # result = [raw, restored]

        # if gt_file is not None:
        #     print("Loading ground truth image from", gt_file)
        #     gt = tifffile.imread(str(gt_file))
        #     if raw.shape == gt.shape:
        #         gt = normalize(gt, args.p_min, args.p_max)
        #         if args.rescale:
        #             restored = rescale(restored, gt)
        #             result = [raw, restored, gt]
        #         else:
        #             result = [raw, restored, gt]
        #     else:
        #         print("Ground truth image discarded due to image shape mismatch")


        if args.normalize_output_range_between_zero_and_one:
            # result = [normalize_between_zero_and_one(m) for m in result]
            restored = normalize_between_zero_and_one(restored)

        # result = np.stack(result)

        # Convert bit depth if requested
        if args.bpp == 8:
            # result = np.clip(255 * result, 0, 255).astype("uint8")
            result = np.clip(255 * restored, 0, 255).astype("uint8")
        elif args.bpp == 16:
            # result = np.clip(65535 * result, 0, 65535).astype("uint16")
            result = np.clip(65535 * restored, 0, 65535).astype("uint16")
        else:
            # keep float32
            result = restored.astype("float32", copy=False)

        # Output file path
        if output_path.is_dir():
            output_file = output_path / raw_file.name
        else:
            output_file = output_path

        print("Saving output image to", output_file)
        # save_tiff(str(output_file), result, args.output_tiff_format)
        save_restored_tiff(str(output_file), result, fmt=args.output_tiff_format)


if __name__ == "__main__":
    main()
