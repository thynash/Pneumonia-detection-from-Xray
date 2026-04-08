import os
import pandas as pd

def get_distribution(base_path="dataset"):
    data = []

    for split in ["train", "val", "test"]:
        for cls in ["normal", "pneumonia"]:
            path = os.path.join(base_path, split, cls)
            count = len(os.listdir(path))
            data.append([split, cls, count])

    df = pd.DataFrame(data, columns=["Split", "Class", "Count"])
    return df


if __name__ == "__main__":
    df = get_distribution()
    print(df)
