import os
import cv2
import random
import matplotlib.pyplot as plt

def show_samples(split, cls, base_path="dataset", n=5):
    path = os.path.join(base_path, split, cls)
    files = random.sample(os.listdir(path), n)

    plt.figure(figsize=(15,3))
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(cls)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    show_samples("train", "normal")
    show_samples("train", "pneumonia")
