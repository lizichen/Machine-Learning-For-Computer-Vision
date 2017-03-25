## Hand Recognition

### Video Files Link (Dropbox Public Link):
    goo.gl/0UMRzk

### To Retrieve Sequence of Images from the Video:
```sh
    ffmpeg -i NYU/HR/Video_Dim/ch01_20170317124730.mp4 -r 12 ch01_%06d.png  
```
option **-r** follows the frequency. 12 means 12 FPS.   
This script will retrieve 8682 PNGs.

### Labels File (in 12 FPS)
Given a csv file **TimeAndLabels_Light_ch01.csv** that marks the timestamp for the beginning of each *'Label'(Hand Gesture)*, and use **MapTimeToImageId.py** to generate another csv file **Light_Ch01_Labels.csv** that has the pairs of {ImageFileName and LabelName}, which can be leveraged when training and testing.

The Light_Ch01_Labels.csv looks like this:
```csv
ch01_000001, N/A
ch01_000641, TWO
ch01_008123, THREE(US)
...
```

There will be some inaccuracy in the labeling when FPS is higher.

**Todo**:  
2. Light_Ch02_Labels.csv  
3. Dim_Ch01_Labels.csv  
4. Dim_Ch02_Labels.csv


