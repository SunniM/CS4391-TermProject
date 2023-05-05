# CS 4391 - Scene Recognition: Report

## About the Data 

 - Images are already split into train and test sets
 - Relatively small resolutions (~250 x 250)
 - Mostly grayscale images
 - Three image classes:
   * Bedroom
   * Coast
   * Forest
## Preprocessing
 1. Every image is explicitly converted into grayscale
 2. The images are resized to (50 x 50) and (200 x 200) and saved in new folders (small) and (large) respectively. Still waiting the test train split.

Old file structure:
```
.\data\
    |
    |--Train
    |    |--Bedroom
    |    |--Coast
    |    |--Forest
    |
    |--Test
    |    |--Bedroom
    |    |--Coast
    |    |--Forest
```
New file structure:
```
.\data\
    |--Small
    |    |--Train
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
    |    |
    |    |--Test
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
    |
    |--Large
    |    |--Train
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
    |    |
    |    |--Test
    |    |    |--Bedroom
    |    |    |--Coast
    |    |    |--Forest
```


## Sift Extraction
In the function 'siftExtract()' is where the SIFT features that are used are extracted and stored in a pickle file is done. As seen in the diagram above some looping was needed in order to acces all the folders that contained the images. While looping through I used the sift.detectandcompute method available in open cv. I then stored the features in a list in order to sort through them and make them the all the same length, padding the ones that are too small with zeros. I then stored those descriptors along with the image scene in the pickle file, one for the test images and one for the train images.
### Nearest Neighbor classifier
The function 'knnClass()' uses the sift features to train and test the Nearest Neighbor Classifier. It first extracts the features stored in the pickle file mentioned above, and reshapes the array so that it can be used in the sklearn's function. After the prep on the data is complete we then create an instance of KNeighborsClassifier called knn. Then the training data along with the training label are fitted into the model using 'knn.fit()' After which it is then fed the test image's data in order to predict which scene it falls under. I then pass the confusion matrix from the predicted results to a function to print the accuracy, False negative, and false positive percentages. the results for ours are as follow: Accuracy: 39.23%,  False Positive: 30.06%, False Negative: 2.49%. Suprisingly low false negative percentages.
### linear SVM classifier
This was implemented in the same way as the Nearest Neighbor classifier mentioned above. First the data was loaded from the pickle files and then processed. It was then fed into the svm model provided from the sklearn library. Then it was tested on the testing image's features, the confusion was then fed to the 'printResults()' function. The results for ours are as follow: Accuracy: 47.84%,  False Positive: 7.98%, False Negative: 22.53%.  These are higher in terms of accuracy and has a lower false percentage then NNC. 




 
