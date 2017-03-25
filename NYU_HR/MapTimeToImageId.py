import csv
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--fps",
        help="Number of frames per second that you extracted from the video.",
        type=int, default=12)

parser.add_argument("-i", "--input",
        help="Path to the CSV file containing the labels and timestamps.",
        default="TimeAndLabels_Light_ch01.csv")

parser.add_argument("-o", "--output",
        help="Name of the output CSV file you would like to create.",
        default = "Light_Ch01_Labels.csv")

parser.add_argument("-p", "--prefix",
        help="Prefix of the image files.",
        default="ch01")

args = parser.parse_args()

print(args)

fps = args.fps
FMT = '%H:%M:%S'
CSV_TIME_LABELS_SRC = args.input
IMAGENAME_LABELS_TARGET = args.output
Camera_Channel = args.prefix

timestamp = []
label = []
with open(CSV_TIME_LABELS_SRC, "rb") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for i, line in enumerate(reader):
        timestamp.append(line[0])
        label.append(line[1])

i = len(timestamp)
timestamp.append(timestamp[i-1])

target = open(IMAGENAME_LABELS_TARGET, 'w+')
write_count = 1 # close and reopen after 8000

img_indexStart = 0
img_indexEnd = 1

for timeindex in range(0, i):
    currenttime = timestamp[timeindex]
    nexttime = timestamp[timeindex+1]
    currentlabel = label[timeindex]
    
    diff_sec = (datetime.strptime(nexttime, FMT) - datetime.strptime(currenttime, FMT)).seconds
    
    img_indexStart = img_indexEnd
    img_indexEnd = img_indexStart + diff_sec * fps
    
    print "Generating {Image_ID, Label} pairs from:"+str(img_indexStart)+" to:"+str(img_indexEnd-1)+" labeling:"+currentlabel
    for rep in range(img_indexStart, img_indexEnd):
        line = Camera_Channel+"_"+'{:06d}'.format(rep) + "," + currentlabel 
        target.write(line)
        target.write("\n")
    
    write_count += 1
    if write_count % 7000 == 0:
        target.close()
        target = open(IMAGENAME_LABELS_TARGET, 'a+')

target.close()
print "done."
