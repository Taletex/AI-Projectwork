### STEP 5: UCF-101-INV VIDEO CREATION ###
# Script usato per creare i video a partire dalle frame estratte da UCF-101 e invertite rispetto alla razza bianca/nera.
# Viene utilizzato ffmpeg tramite comando lanciato da terminale dallo script.
### --------------------------------------- ###

import os
import sys

def main():
    baseSourceDir = "../datasets/UCF-101-Inv/train"
    baseDestDir = "../datasets/UCF-101-Inv-Video"

    # creating destination directory
    if not os.path.exists(baseDestDir):
        os.mkdir(baseDestDir)

    baseDestDir = baseDestDir + "/train"
    if not os.path.exists(baseDestDir):
        os.mkdir(baseDestDir)

    # for each class
    for extIndex, videoDir in enumerate(os.listdir(baseSourceDir)):
        print(f"Elaborazione: {videoDir}")

        sourceDir = baseSourceDir + "/" + videoDir
        destDir = baseDestDir + "/" + videoDir
        if not os.path.exists(destDir):
            os.mkdir(destDir)

        # For each video folder in source video folder
        for innIndex, folder in enumerate(os.listdir(sourceDir)):
            sourceVideoDir = sourceDir + "/" + folder

            cmd = "ffmpeg -r 25 -f image2 -i "+ sourceVideoDir +"/img" + folder + "_%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p " + destDir + "/" + folder + ".avi > /dev/null 2>&1"
            os.system(cmd)

# Driver Code
if __name__ == '__main__':
    main()
