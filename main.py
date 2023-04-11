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
                    new_path = os.path.join("new_data", s, set_type.lower(), class_type.lower())
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    cv.imwrite(os.path.join(new_path, file), resized)
    return



def  siftExtract():
    fieldNames = ['Category', 'Keypoints', 'descriptors']
    with open('sift_Ft.csv', 'w', encoding='UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames= fieldNames)
        writer.writeheader()
    
    csvFile = open('sift_Ft.csv', 'w')
    writer = csv.writer(csvFile, delimiter='-')

    sift = cv.SIFT_create()
    set_type2 = "train"
    for set_type in ["large", "small"]:
        set_path1 = os.path.join("ProcData", set_type, set_type2)
        for folderName in os.listdir(set_path1):
            folderPath = os.path.join(set_path1, folderName)
            for file in os.listdir(folderPath):
                file_path = os.path.join(folderPath, file)
                img  = cv.imread(file_path)
                keypoints, descriptors = sift.detectAndCompute(img, None)
                row = [f'{folderName}', keypoints, descriptors]
                writer.writerow(row)

    csvFile.close()
    
    

def main():
    if not os.path.exists("new_data"):
        preprocess()
        siftExtract()









if __name__ == "__main__":
    main()
