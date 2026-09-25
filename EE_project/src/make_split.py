import os
import argparse
import random
from collections import defaultdict

import numpy as np
import scipy.io as sio
import h5py

# ============================================================
# Config
# ============================================================

mode = 'r'
if mode == 'r':
    DEFAULT_YS_DIR = "crop_S_r"         # real Sentinel-2
    DEFAULT_YH_DIR = "crop_A_r"           # AVIRIS
    DEFAULT_OUT_DIR = "split_real"
else:
    DEFAULT_YS_DIR = "crop_S"         # real Sentinel-2
    DEFAULT_YH_DIR = "crop_A"           # AVIRIS
    DEFAULT_OUT_DIR = "split_simulation"


# MATLAB .mat source_file reader
def decode_char_array(data):
    """
    Decode MATLAB char / HDF5 char array into Python string.
    """
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore").strip()

    data = np.array(data)

    if np.issubdtype(data.dtype, np.integer):
        flat = data.flatten()
        chars = [chr(int(x)) for x in flat if int(x) != 0]
        return "".join(chars).strip()

    if data.dtype.kind == "S":
        return b"".join(data.flatten()).decode("utf-8", errors="ignore").strip()

    if data.dtype.kind == "U":
        return "".join(data.flatten().astype(str)).strip()

    return str(data).strip()


def decode_hdf5_string(obj, h5file):
    data = obj[()]

    if isinstance(data, h5py.Reference):
        data = h5file[data][()]

    if isinstance(data, np.ndarray) and data.dtype == object:
        chars = []
        for ref in data.flatten():
            if isinstance(ref, h5py.Reference):
                ref_data = h5file[ref][()]
                chars.append(decode_char_array(ref_data))
        return "".join(chars).strip()

    return decode_char_array(data)


def read_source_file(mat_path):
    """
    Strictly read source_file from .mat.
    Supports MATLAB v7 and v7.3.
    Returns a Python string.
    Raises error if source_file is missing or unreadable.
    """

    # Try scipy first
    try:
        mat = sio.loadmat(mat_path)

        if "source_file" not in mat:
            raise KeyError(f"source_file not found in {mat_path}")

        src = mat["source_file"]

        if isinstance(src, str):
            out = src.strip()
        elif src.size == 1:
            item = src.item()
            if isinstance(item, bytes):
                out = item.decode("utf-8", errors="ignore").strip()
            else:
                out = str(item).strip()
        elif hasattr(src, "dtype") and src.dtype.kind in ["U", "S"]:
            out = "".join(src.flatten().astype(str)).strip()
        else:
            out = str(src).strip()

        if out == "":
            raise ValueError(f"source_file is empty in {mat_path}")

        return out

    except NotImplementedError:
        # MATLAB v7.3; use h5py below
        pass

    # Try h5py for MATLAB v7.3
    with h5py.File(mat_path, "r") as f:
        if "source_file" not in f.keys():
            raise KeyError(f"source_file not found in {mat_path}")

        out = decode_hdf5_string(f["source_file"], f).strip()

        if out == "":
            raise ValueError(f"source_file is empty in {mat_path}")

        return out

def list_mat_files(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    return {
        f for f in os.listdir(folder)
        if f.lower().endswith(".mat")}


def get_common_files(ys_dir, yh_dir):
    ys_files = list_mat_files(ys_dir)
    yh_files = list_mat_files(yh_dir)

    common = sorted(list(ys_files & yh_files))

    missing_in_ys = sorted(list(yh_files - ys_files))
    missing_in_yh = sorted(list(ys_files - yh_files))

    print("[File check]")
    print(f"YS folder: {ys_dir}")
    print(f"YH folder: {yh_dir}")
    print(f"YS .mat files: {len(ys_files)}")
    print(f"YH .mat files: {len(yh_files)}")
    print(f"Common files: {len(common)}")
    print(f"YH files missing in YS: {len(missing_in_ys)}")
    print(f"YS files missing in YH: {len(missing_in_yh)}")

    if len(common) == 0:
        raise RuntimeError("No common .mat files found between crop_S and crop_A.")

    return common, missing_in_ys, missing_in_yh


def build_source_groups(ys_dir, yh_dir, common_files):
    """
    Group patch files by source_file.
    Strict rule:
    - source_file must exist in both crop_S and crop_A.
    - source_file must match between crop_S and crop_A.
    """
    source_to_files = defaultdict(list)

    for fname in common_files:
        ys_path = os.path.join(ys_dir, fname)
        yh_path = os.path.join(yh_dir, fname)

        src_ys = read_source_file(ys_path)
        src_yh = read_source_file(yh_path)

        if src_ys != src_yh:
            raise RuntimeError(
                f"source_file mismatch in {fname}\n"
                f"  crop_S source_file: {src_ys}\n"
                f"  crop_A source_file: {src_yh}"
            )

        source_to_files[src_ys].append(fname)

    return source_to_files


def split_sources(source_list, train_ratio, val_ratio, seed):
    """
    Split by source_file, not by patch file.
    """
    source_list = list(source_list)

    if len(source_list) < 3:
        raise RuntimeError(
            f"Need at least 3 source_file groups for train/val/test split. "
            f"Got {len(source_list)}.")

    rng = random.Random(seed)
    rng.shuffle(source_list)

    n = len(source_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    n_train = max(1, n_train)
    n_val = max(1, n_val)

    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    train_sources = source_list[:n_train]
    val_sources = source_list[n_train:n_train + n_val]
    test_sources = source_list[n_train + n_val:]

    return train_sources, val_sources, test_sources


def expand_files(source_to_files, sources):
    files = []

    for src in sources:
        files.extend(source_to_files[src])

    return sorted(files)


def save_txt(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(str(item) + "\n")


def check_no_overlap(a, b, name_a, name_b):
    overlap = set(a) & set(b)

    print(f"{name_a} ∩ {name_b}: {len(overlap)}")

    if len(overlap) > 0:
        examples = sorted(list(overlap))[:10]
        raise RuntimeError(
            f"Overlap detected between {name_a} and {name_b}. "
            f"Examples: {examples}"
        )


def print_source_summary(source_to_files):
    print("\n[Source summary]")
    for src in sorted(source_to_files.keys()):
        files = source_to_files[src]
        print(f"{src}: {len(files)} patches")



# Main
def main():
    parser = argparse.ArgumentParser(
        description="Create source-level train/val/test split for real Sentinel-2 COS2A."
    )

    parser.add_argument("--ys_dir", type=str, default=DEFAULT_YS_DIR)
    parser.add_argument("--yh_dir", type=str, default=DEFAULT_YH_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("====================================")
    print("Real Sentinel-2 source-level split")
    print("====================================")
    print(f"YS dir  : {args.ys_dir}")
    print(f"YH dir  : {args.yh_dir}")
    print(f"Out dir : {args.out_dir}")
    print(f"Ratio   : train={args.train_ratio}, val={args.val_ratio}, test=remaining")
    print(f"Seed    : {args.seed}")

    common_files, missing_in_ys, missing_in_yh = get_common_files(args.ys_dir, args.yh_dir)

    if len(missing_in_ys) > 0:
        save_txt(missing_in_ys, os.path.join(args.out_dir, "missing_in_ys.txt"))

    if len(missing_in_yh) > 0:
        save_txt(missing_in_yh, os.path.join(args.out_dir, "missing_in_yh.txt"))

    source_to_files = build_source_groups(
        args.ys_dir,
        args.yh_dir,
        common_files)

    source_list = sorted(source_to_files.keys())

    print("\n[Source check]")
    print(f"Unique source_file groups: {len(source_list)}")

    print_source_summary(source_to_files)

    train_sources, val_sources, test_sources = split_sources(
        source_list,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed)

    print("\n[Source overlap check]")
    check_no_overlap(train_sources, val_sources, "train_sources", "val_sources")
    check_no_overlap(train_sources, test_sources, "train_sources", "test_sources")
    check_no_overlap(val_sources, test_sources, "val_sources", "test_sources")

    train_files = expand_files(source_to_files, train_sources)
    val_files = expand_files(source_to_files, val_sources)
    test_files = expand_files(source_to_files, test_sources)

    print("\n[Patch overlap check]")
    check_no_overlap(train_files, val_files, "train_files", "val_files")
    check_no_overlap(train_files, test_files, "train_files", "test_files")
    check_no_overlap(val_files, test_files, "val_files", "test_files")

    save_txt(train_files, os.path.join(args.out_dir, "train.txt"))
    save_txt(val_files, os.path.join(args.out_dir, "val.txt"))
    save_txt(test_files, os.path.join(args.out_dir, "test.txt"))

    save_txt(train_sources, os.path.join(args.out_dir, "train_sources.txt"))
    save_txt(val_sources, os.path.join(args.out_dir, "val_sources.txt"))
    save_txt(test_sources, os.path.join(args.out_dir, "test_sources.txt"))

    print("\n[Split result]")
    print(f"Train sources: {len(train_sources)}")
    print(f"Val sources  : {len(val_sources)}")
    print(f"Test sources : {len(test_sources)}")

    print(f"Train patches: {len(train_files)}")
    print(f"Val patches  : {len(val_files)}")
    print(f"Test patches : {len(test_files)}")

    print("\n[Examples]")
    print("Train:", train_files[:5])
    print("Val  :", val_files[:5])
    print("Test :", test_files[:5])

    print("\nSaved split files to:", args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()