import os
import cv2 as cv
import csv
import numpy as np
import pandas as pd

LABEL = {'bedroom': 0, 'coast': 1, 'forest': 2}
SIZES = {"small": (50, 50), "large": (200, 200)}


def preprocess():
    for set_type in ["Test", "Train"]:
        set_path = os.path.join("data", set_type)
        for class_type in os.listdir(set_path):
            class_path = os.path.join(set_path, class_type)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                if (img is None):
                    print("Image Not Found")
                    exit(0)
                for size, shape in SIZES.items():
                    resized = cv.resize(img, shape)
                    new_path = os.path.join(
                        "new_data", size, set_type.lower(), class_type.lower())
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    cv.imwrite(os.path.join(new_path, file), resized)
    return


def siftExtract():
    fieldNames = ['Category', 'Keypoints', 'descriptors']
    with open('sift_Ft.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldNames)
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
                img = cv.imread(file_path)
                if (img is None):
                    print(f'Image: {file_path} Not Found')
                    exit(0)
                keypoints, descriptors = sift.detectAndCompute(img, None)
                row = [f'{folderName}', keypoints, descriptors]
                writer.writerow(row)

    csvFile.close()


def histExtract():
    
    fieldnames  = [x for x in range(256)]
    fieldnames.append('Y')

    set_type = ['train', 'test']
    for type in set_type:
        with open(f'hist_{type}.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for size in ["large", "small"]:
        
        size_path = os.path.join("new_data", size)
        for set_type in os.listdir(size_path):

            csvFile = open(f'hist_{set_type}.csv', 'a')
            writer = csv.writer(csvFile, delimiter=',', lineterminator='\n')

            set_path = os.path.join(size_path, set_type)
            for class_type in os.listdir(set_path):
                
                class_path = os.path.join(set_path, class_type)
                i = 0
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

    train = pd.read_csv('hist_train.csv')
    train_label = train.pop('Y').to_numpy(np.int32)
    train = train.to_numpy(np.float32)

    test = pd.read_csv('hist_test.csv')
    test_label = test.pop('Y').to_numpy(np.int32)
    test = test.to_numpy(np.float32)
    return train, train_label, test, test_label


def imgExtract(size):

    for i, set_type in enumerate(["train", "test"]):
        examples, labels = None, None
        set_path = os.path.join("new_data",size, set_type)
        for class_type in os.listdir(set_path):
            class_path = os.path.join(set_path, class_type)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                if(img is None):
                    print("Image Not Found")
                    exit(0)
                img = img.flatten()
                if(examples is None):
                    examples = img.astype(np.float32)
                    labels = np.asarray(LABEL[class_type])
                else:
                    examples = np.vstack([examples, img.astype(np.float32)])
                    labels = np.append(labels, LABEL[class_type])
                
        if i == 0:
            train = examples
            train_label = labels
        else:
            test = examples
            test_label = labels
    return train, train_label, test, test_label
   

def nearestNeighbor(train, train_label, test):
    
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_label)
    _, results, _, _ = knn.findNearest(test, 3)
    return results

def printSummary(actual, pred):
    print()
    confusion = np.zeros((3,3))
    for i in range(len(actual)):
        confusion[actual[i]][pred[i]] += 1
    
    acc = 0
    for i in range(3):
        acc += confusion[i][i]
    acc = acc/len(actual)
    print('accuracy: %.3f' % acc)
    
    confusion = pd.DataFrame(confusion, columns=list(LABEL.keys()), index=list(LABEL.keys()), dtype='int32')
    print(confusion)
    print()




def main():
    if not os.path.exists("new_data"):
        preprocess()

    train, train_label, test, test_label = imgExtract('small')
    pred = nearestNeighbor(train, train_label, test)
    pred = pred.reshape(pred.shape[0],)
    pred = pred.astype(np.int32)

    print('KNN using pixel values:')
    printSummary(test_label, pred)

    siftExtract()
    
    train, train_label, test, test_label = histExtract() 
    pred = nearestNeighbor(train, train_label, test)
    pred = pred.reshape(pred.shape[0],)
    pred = pred.astype(np.int32)

    print('KNN using histogram features:')
    printSummary(test_label, pred)

if __name__ == "__main__":
    main()
