### STEPS 6, 7, 8: VIDEO CLASSIFICATION TASK ###
# Script usato per creare i file .txt usati dal video classifier per generare l'annotation file .json.
# Questi file txt indicano per ogni video, se questo appartiene al test set o al training set.
### --------------------------------------- ###

import os
import sys
import pandas as pd

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = {}
    for i in range(data.shape[0]):
        labels.update({data.ix[i, 1] : data.ix[i, 0]})

    return labels

def main():
    folders = {0: "Splitted", 1: "Inv-Video-Splitted", 2: "Mrg-Splitted"}

    #una volta UCF-101-Inv, una volta UCF-101-splitted
    for j in range(0, 3):
        baseSourceDir = "../../../../../media/alessandro/HDD500GB/datasets/UCF-101-" + folders[j] #baseSourceDir = "../../../../media/alessandro/HDD500GB/datasets/UCF-101-Inv"
        baseDestDir = "metadata/" + folders[j]
        actions = {0: "train", 1: "test"}
        labels = load_labels("metadata/classInd.txt")

        if not os.path.exists(baseDestDir):
            os.makedirs(baseDestDir)

        for x in range(0, 2):
            nameList = []
            sourceDir = baseSourceDir + "/" + actions[x]
            destFile = baseDestDir + "/" + actions[x] + "list0"
            # for each class
            for extIndex, videoDir in enumerate(os.listdir(sourceDir)):
                sourceVideoDir = sourceDir + "/" + videoDir

                # For each video in source video folder
                for innIndex, name in enumerate(os.listdir(sourceVideoDir)):
                    fileName = videoDir + "/" + name + " " + str(labels[videoDir])
                    nameList.append(fileName)

            nameList.sort()

            for k in range(1, 4):
                dst_file = open(destFile + str(k) + ".txt", "w+")
                for idx, elem in enumerate(nameList):
                    dst_file.write(elem + "\n")

        cmd = "python ../../3D-ResNets-PyTorch/utils/ucf101_json.py metadata/" + folders[j]
        os.system(cmd)

# Driver Code
if __name__ == '__main__':
    main()
