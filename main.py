import os
import cv2 as cv

def preprocess():
    sizes = {"small": (50,50), "large" : (200,200)}
    for set_type in ["Test", "Train"]:
        set_path = os.path.join("data", set_type)
        for class_type in os.listdir(set_path):
            class_path = os.path.join(set_path, class_type)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                for s in sizes:
                    resized = cv.resize(img, sizes[s])
                    new_path = os.path.join("new_data", s, set_type.lower(), class_type.lower(), file)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    cv.imwrite(os.path.join(new_path, file), resized)
    return

def main():
    if not os.path.exists("new_data"):
        preprocess()










if __name__ == "__main__":
    main()