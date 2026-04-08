import os
import cv2
import pandas as pd

def get_image_sizes(base_path="dataset"):
    sizes = []

    for cls in ["normal", "pneumonia"]:
        path = os.path.join(base_path, "train", cls)

        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            sizes.append(img.shape[:2])

    df = pd.DataFrame(sizes, columns=["Height", "Width"])
    return df


if __name__ == "__main__":
    df = get_image_sizes()
    print(df.describe())
