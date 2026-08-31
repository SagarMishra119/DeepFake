import os
import random
import shutil

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(BASE_DIR, "dataset", "real_vs_fake", "real-vs-fake")
dest = os.path.join(BASE_DIR, "processed")

sample_targets = {
    "train": 2800,
    "valid": 600,
    "test": 600,
}

classes = ["real", "fake"]
valid_extensions = (".jpg", ".jpeg")


def sample_and_copy(split, cls, n_samples):
    """
    Sample n_samples images from source/split/cls and copy them
    into dest/split/cls. Returns number of files copied.
    """
    source_folder = os.path.join(source, split, cls)
    dest_folder = os.path.join(dest, split, cls)

    os.makedirs(dest_folder, exist_ok=True)

    all_files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(valid_extensions)
    ]

    if len(all_files) < n_samples:
        raise ValueError(
            f"Requested {n_samples} samples from {source_folder}, "
            f"but only {len(all_files)} available."
        )

    random.shuffle(all_files)
    sampled_files = all_files[:n_samples]

    copied_count = 0
    for filename in sampled_files:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(dest_folder, filename)
        shutil.copy2(src_path, dst_path)
        copied_count += 1

    return copied_count


def verify_output():
    print("\n--- Verification ---")
    for split, target_per_class in sample_targets.items():
        for cls in classes:
            dest_folder = os.path.join(dest, split, cls)
            count = len([
                f for f in os.listdir(dest_folder)
                if f.lower().endswith(valid_extensions)
            ])
            status = "OK" if count == target_per_class else "MISMATCH"
            print(f"{split}/{cls}: {count} files (expected {target_per_class}) [{status}]")


def main():
    print("Starting sampling...\n")
    for split, n in sample_targets.items():
        for cls in classes:
            copied = sample_and_copy(split, cls, n)
            print(f"Copied {copied} files -> {split}/{cls}")

    verify_output()
    print("\nSampling complete.")


if __name__ == "__main__":
    main()
