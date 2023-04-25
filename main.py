import os
import cv2 as cv
import csv
import numpy as np
import pandas as pd

LABEL = {'bedroom':0, 'coast':1, 'forest':2}

def preprocess():
    sizes = {"small": (50,50), "large" : (200,200)}
    for set_type in ["Test", "Train"]:
        set_path = os.path.join("data", set_type)
        for class_type in os.listdir(set_path):
            class_path = os.path.join(set_path, class_type)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                if(img is None):
                    print("Image Not Found")
                    exit(0)
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
        set_path = os.path.join("new_data", set_type, set_type2)
        for folderName in os.listdir(set_path):
            folderPath = os.path.join(set_path, folderName)
            for file in os.listdir(folderPath):
                file_path = os.path.join(folderPath, file)
                img  = cv.imread(file_path)
                if(img is None):
                    print(f'Image: {file_path} Not Found')
                    exit(0)
                keypoints, descriptors = sift.detectAndCompute(img, None)
                row = [f'{folderName}', keypoints, descriptors]
                writer.writerow(row)

    csvFile.close()
    
def histExtract():
    fieldNames  = [x for x in range(256)]
    fieldNames.append('Y')

    for size in ["large", "small"]:
        
        size_path = os.path.join("new_data", size)
        for set_type in os.listdir(size_path):
            
            with open(f'hist_{set_type}.csv', 'w', encoding='UTF8', newline = '') as f:
                writer = csv.DictWriter(f, fieldnames=fieldNames)
                writer.writeheader()
            csvFile = open(f'hist_{set_type}.csv', 'a')
            writer = csv.writer(csvFile, delimiter=',')

            set_path = os.path.join(size_path, set_type)
            for class_type in os.listdir(set_path):
                
                class_path = os.path.join(set_path, class_type)
                for file in os.listdir(class_path):
                    file_path = os.path.join(class_path, file)
                    img  = cv.imread(file_path)
                    if(img is None):
                        print(f'Image: {file_path} Not Found')
                        exit(0)

                    histSize = 256
                    histRange = (0, 256) # the upper boundary is exclusive
                    accumulate = False
                    hist = cv.calcHist(img, [0], None, [histSize], histRange, accumulate=accumulate)
                    hist = hist.reshape((256,))
                    hist = np.append(hist,LABEL[class_type])
                    writer.writerow(hist.astype(np.float32))
            csvFile.close()

def nearestNeighbor(train,train_label, test, test_label):
    knn = cv.ml.KNearest_create()

    knn.train(train, cv.ml.ROW_SAMPLE, train_label)

    ret, results,neighbours,dist = knn.findNearest(test, 3)
    print( "result:  {}\n".format(results) )
    print( "neighbours:  {}\n".format(neighbours) )
    print( "distance:  {}\n".format(dist) )



def main():
    if not os.path.exists("new_data"):
        preprocess()
    siftExtract()
    
    histExtract()
    train = pd.read_csv('hist_train.csv')
    train_label = train.pop('Y').to_numpy(np.float32)
    train = train.to_numpy(np.float32)

    test = pd.read_csv('hist_test.csv')
    test_label = test.pop('Y').to_numpy(np.float32)
    test = test.to_numpy(np.float32)

    nearestNeighbor(train, train_label, test, test_label)








if __name__ == "__main__":
    main()
