from __future__ import division
import sys
import torchvision.transforms as T
import cv2
import time
import numpy
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import PIL
sys.path.insert(0, '../../pytorch-CycleGAN-and-pix2pix')
import test_custom
class CNN(nn.Module):

    # Constructor
    def __init__(self):
        # Call parent constructor
        super().__init__();
        # Create convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # Create fully-connected layers
        self.fc_layers = nn.Sequential(
            # FC layer
            nn.Linear(4096, 1024),
            nn.ReLU(),
            # Classification layer
            nn.Linear(1024, 2)
        )

    # Forward
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
import face_classification

def detectFaceOpenCVDnn(net, frame, index, frame_count, destDir):
    saveName = destDir + "/img" + str(index) + "_" + str(frame_count) + ".jpg"
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    # convert the image from the current frame
    img = Image.fromarray(cv2.cvtColor(numpy.array(frame), cv2.COLOR_RGB2BGR))

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)

            # crop and save the face image from the current frame
            croppedImg = T.functional.crop(img, y1, x1, (y2-y1), (x2-x1))
            croppedImg.save("./img.jpg","JPEG", icc_profile=img.info.get('icc_profile'))
            saveName = destDir + "/img" + str(index) + "_" + str(frame_count) + "_F.jpg"

            # save in the CycleGAN test folders
            croppedImg.save("../../pytorch-CycleGAN-and-pix2pix/datasets/UCF-101/testA/img.jpg","JPEG", icc_profile=img.info.get('icc_profile'))
            croppedImg.save("../../pytorch-CycleGAN-and-pix2pix/datasets/UCF-101/testB/img.jpg","JPEG", icc_profile=img.info.get('icc_profile'))

            # execute the CycleGAN
            test_custom.executeCyclegan()

            # delete the saved images
            os.remove("../../pytorch-CycleGAN-and-pix2pix/datasets/UCF-101/testA/img.jpg")
            os.remove("../../pytorch-CycleGAN-and-pix2pix/datasets/UCF-101/testB/img.jpg")

            # Classification and loading of the right cyclegan image
            if(face_classification.face_classifier("./img.jpg") == 0):
                cycleGanImg = Image.open("./results/blackwhite_cyclegan/test_latest/images/img_fake_B.png")
            else:
                cycleGanImg = Image.open("./results/blackwhite_cyclegan/test_latest/images/img_fake_A.png")

            # paste the img in the frame (need to resize the img first)
            cycleGanImg = cycleGanImg.resize(((x2-x1), (y2-y1)), PIL.Image.ANTIALIAS)
            img.paste(cycleGanImg, (x1, y1))

    # save the current frame
    img.save(saveName,"JPEG", icc_profile=img.info.get('icc_profile')) # img0_123_S2

    return frameOpencvDnn, bboxes

if __name__ == "__main__" :
    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    DNN = "TF"
    if DNN == "CAFFE":
        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    source = 0
    sourceDir = "../../AI-Projectwork/datasets/UCF-101-unfolded/a"
    destDir = "UCF-101-Inv"

    # Create dest images folder
    if not os.path.exists(destDir):
        os.makedirs(destDir)

    # Create folder for CycleGan computation
    if not os.path.exists(destDir):
        os.makedirs(destDir)

    # For each video in the arg folder
    for index, filename in enumerate(os.listdir(sourceDir)):

        source = sourceDir + "/" + filename
        cap = cv2.VideoCapture(source)
        hasFrame, frame = cap.read()

        vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

        frame_count = 0
        tt_opencvDnn = 0
        while(1):

            hasFrame, frame = cap.read()
            if not hasFrame:
                break
            frame_count += 1

            t = time.time()
            outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame, index, frame_count, destDir)
            tt_opencvDnn += time.time() - t
            fpsOpencvDnn = frame_count / tt_opencvDnn
            label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
            cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("Face Detection Comparison", outOpencvDnn)

            vid_writer.write(outOpencvDnn)
            if frame_count == 1:
                tt_opencvDnn = 0

            k = cv2.waitKey(10)
            if k == 27:
                break
        cv2.destroyAllWindows()
        vid_writer.release()
