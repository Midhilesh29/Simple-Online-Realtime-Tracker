import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

class Object_Detector():
    def __init__(self):
        self.confThreshold=0.5
        self.nmsThreshold=0.4

        self.inpWidth=416
        self.inpHeight=416

        self.classesFile="/home/midhilesh/Desktop/IIIT B research/YOLO object tracking/coco.names"

        self.classes=None

        with open(self.classesFile,'rt') as f:
            self.classes=f.read().rstrip('\n').split('\n')
        
        self.modelConfiguration="/home/midhilesh/Desktop/IIIT B research/YOLO object tracking/yolov3.cfg"
        self.modelWeights="/home/midhilesh/Desktop/IIIT B research/YOLO object tracking/yolov3.weights"

        self.net=cv.dnn.readNetFromDarknet(self.modelConfiguration,self.modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def getOutputNames(self,net):
        layersNames=self.net.getLayerNames()
        return[layersNames[i[0]-1] for i in self.net.getUnconnectedOutLayers()]

    def postprocess(self,frame,outs):
        frameHeight=frame.shape[0]
        frameWidth=frame.shape[1]

        classIds=[]
        confidences=[]
        boxes=[]

        for out in outs:
            for detection in out:
                scores=detection[5:]
                classId=np.argmax(scores)
                confidence=scores[classId]

                if(confidence>self.confThreshold):
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        NewBox=[]
        NewClass=[]
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            NewClass.append(self.classes[classIds[i]])
            #add the new fuction here
            NewBox.append((x,y,w,h))
        return [NewBox,NewClass]


    def localization(self,frame):
        blob=cv.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outs=self.net.forward(self.getOutputNames(self.net))
        [BoundingBox,Pred_class]=self.postprocess(frame, outs)
        return [BoundingBox,Pred_class]
