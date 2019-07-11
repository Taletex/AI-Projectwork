import os

def main():
    baseSourceDir = "../datasets/UCF-101-Inv/train"

    # for each class
    for extIndex, videoDir in enumerate(os.listdir(baseSourceDir)):
        sourceDir = baseSourceDir + "/" + videoDir

        # For each video folder in source video folder
        for innIndex, folder in enumerate(os.listdir(sourceDir)):
            sourceVideoDir = sourceDir + "/" + folder

            # for each video
            for index, filename in enumerate(os.listdir(sourceVideoDir)):
                newfilename = filename.replace("_F", "", 1)
                dst = sourceVideoDir + "/" + newfilename
                src = sourceVideoDir + "/" + filename
                # rename() function will
                # rename all the files
                os.rename(src, dst)

# Driver Code
if __name__ == '__main__':

    main()
