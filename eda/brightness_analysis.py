import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def brightness_distribution(base_path="dataset"):
    brightness = []

    for cls in ["normal", "pneumonia"]:
        path = os.path.join(base_path, "train", cls)

        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            brightness.append(np.mean(img))

    return brightness


if __name__ == "__main__":
    values = brightness_distribution()
    plt.hist(values, bins=50)
    plt.title("Brightness Distribution")
    plt.show()
