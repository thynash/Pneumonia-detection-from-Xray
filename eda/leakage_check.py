import os

def check_leakage(base_path="dataset"):
    train_files = set(os.listdir(os.path.join(base_path, "train", "normal")))
    test_files = set(os.listdir(os.path.join(base_path, "test", "normal")))

    overlap = train_files.intersection(test_files)
    return len(overlap)


if __name__ == "__main__":
    print("Overlap:", check_leakage())
