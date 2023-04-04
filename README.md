# CS 4391 - Scene Recognition

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
 
