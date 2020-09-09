from collections import Counter
from uuid import uuid4
import numpy as np
import cv2
from math import atan2
from math import pi
from MqttClient import sendMessage

# this class holds the detected objects and tracks them
class TrafficObject:
    # Class variables
    imgHeight = 0
    imgWidth = 0
    pff = [] # Fitting factors (delivered by main program)
    minPixel = 0 # Earliest/ latest measurement points
    maxPixel = 0 
    framesForSpeedCalc = 0 # last n frames to consider for speed calculation
    labels = [] # Labels for the classes
    allowedClasses = [] # Classes to be considered
    jsonvehicleDataMsg = "" # The template for the MQTT message (filled by config file)
    capabilityAlternateId = ""
    sensorAlternateId = ""
    mqttClient = "" # Will be set at first call by main
    measuresTopicLevel = ""
    sapIotDeviceID = ""
    roiArea = [] #ROI area defined from config file
    trafficUpdown = False #Direction of traffic from config file
    xMeterPixel = 0.


    # Constructor method
    def __init__(self, tTracker, bBox, classLabel, creditLimit, detectionMissDebit, maxCredit, timestamp, detectorNo):
        self.id = str(uuid4()) # assign a unique ID
        self.tracker = tTracker
        self.box = bBox
        self.detectionCredit = creditLimit # start with the limit
        self.creditLimit = creditLimit
        self.labels = [classLabel]
        self.detectionMissDebit = detectionMissDebit
        self.maxCredit = maxCredit
        self.track = []
        self.track.append((timestamp, TrafficObject.__calcCenter(bBox))) # start track
        self.speed = 0.
        self.detectorNo = detectorNo
        self.direction = 0
        self.compassDirection = ""

    # Destructor method
    def __del__(self):
        jsonMsg = self.__createMsg()
        sendMessage(TrafficObject.mqttClient, TrafficObject.measuresTopicLevel+TrafficObject.sapIotDeviceID, jsonMsg) # send the object data to SAP IoT
        print("Object {} says good bye.".format(self.id))
    
    # Calculates the intersection over union (class method)
    def __calcIoU(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    # Create tracker object (class method)
    def createTracker(trackerType):
        if trackerType == 'BOOSTING':
            newTracker = cv2.TrackerBoosting_create()
        if trackerType == 'MIL':
            newTracker = cv2.TrackerMIL_create()
        if trackerType == 'KCF':
            newTracker = cv2.TrackerKCF_create()
        if trackerType == 'TLD':
            newTracker = cv2.TrackerTLD_create()
        if trackerType == 'MEDIANFLOW':
            newTracker = cv2.TrackerMedianFlow_create()
        if trackerType == 'GOTURN':
            newTracker = cv2.TrackerGOTURN_create()
        if trackerType == 'MOSSE':
            newTracker = cv2.TrackerMOSSE_create()
        if trackerType == "CSRT":
            newTracker = cv2.TrackerCSRT_create()
        return newTracker

    # Get class
    def getClassName(classID):
        return TrafficObject.labels[classID]

    # Is class in scope?
    def isClassRelevant(classID):
        return TrafficObject.getClassName(classID) in TrafficObject.allowedClasses
        
    # Get IoU of tracker box vs detection
    def getIoU(self, boxInput):        
        return self.__calcIoU(boxInput, self.__getBBox())

    # Get object ID
    def getId(self):
        return self.id

    # Add detection counter
    def addCount(self, tracker, image, bBox, credit):
        if self.detectionCredit <= (self.creditLimit + self.maxCredit):
            self.detectionCredit += credit
        self.tracker = tracker
        self.tracker.init(image, bBox) #Note! We initialize the tracker to adjust the bounding box size

    # Add label
    def addLabel(self, classLabel):
        self.labels.append(classLabel)

    # Get bounding box of tracker
    def __getBBox(self):
        return self.box

    def __direction_lookup(self, destination_x, origin_x, destination_y, origin_y):
        deltaX = destination_x - origin_x
        deltaY = destination_y - origin_y
        degrees_temp = atan2(deltaX, deltaY)/pi*180
        if degrees_temp < 0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp
        compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        compass_lookup = round(degrees_final / 45)
        return compass_brackets[compass_lookup], degrees_final

    # Get credit factor of tracker (must be greater than zero)
    def getCredit(self):
        creditFactor = int(100 * self.detectionCredit / self.creditLimit)
        if creditFactor < 0:
            creditFactor = 0
        return creditFactor

    # Check if tracker is still within image
    def isGone(self):
        objectIsGone = False
        (cX, cY) = TrafficObject.__calcCenter(self.box)
        relevantRoiArea = TrafficObject.roiArea[4*self.detectorNo-4:4*self.detectorNo-1]
        minX = min(relevantRoiArea, key=lambda item:item[0])[0]
        minY = min(relevantRoiArea, key=lambda item:item[1])[1]
        maxX = max(relevantRoiArea, key=lambda item:item[0])[0]
        maxY = max(relevantRoiArea, key=lambda item:item[1])[1]

        if (cX >= maxX) or (cY >= maxY) or (cX <= minX) or (cY <= minY):
            objectIsGone = True
        return objectIsGone

    # Add center of frame (of tracker) to list
    def __calcCenter(box):
        cX = box[0]+((box[2]-box[0])/2)
        cY = box[1]+((box[3]-box[1])/2)
        return (cX, cY)  

    # Get bounding box absolute coordinates
    def getBBoxCoord(self):
        return (self.box[0], self.box[1]), (self.box[2], self.box[3])

    # Update tracker with new frame
    def updateTracker(self, image, timestamp):
        ok, bbox = self.tracker.update(image)
        self.detectionCredit -= self.detectionMissDebit # We remove the credit from the last detection
        if ok:
            self.box = (int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            
            self.track.append((timestamp, TrafficObject.__calcCenter(self.box))) # extend track
            self.speed = self.__getSpeed(True) # Try to get speed over last 4 measurements
            return True
        else:
            return False #tracker lost track 

    # This checks if tracker is still in boundaries for distance measurement
    def __checkBoundary(self, x,y):
        if TrafficObject.trafficUpdown: #Up/Down traffic
            chkPos = y
        else:                           #Left/Right traffic
            chkPos = x
        if chkPos >= TrafficObject.minPixel and chkPos <= TrafficObject.maxPixel: 
            return True
        else:
            return False

    # Calculate speed
    def __getSpeed(self, useAllData=False, nFrames = 2):
        speed = 0
        n = len(self.track)
        if n>=nFrames: # Check we have sufficient measurements
            if useAllData: # We measure from first trackpoint to current trackpoint (higher precision?)
                for xyP in self.track:
                    x0 = xyP[1][0]
                    y0 = xyP[1][1]                    
                    t0 = xyP[0]                    
                    if self.__checkBoundary(x0,y0): break
                for xyP in reversed(self.track):
                    x1 = xyP[1][0]
                    y1 = xyP[1][1]
                    t1 = xyP[0]
                    if self.__checkBoundary(x1,y1): break
            else:
                x0 = self.track[n-nFrames][1][0]   
                y0 = self.track[n-nFrames][1][1]        
                t0 = self.track[n-nFrames][0]
                x1 = self.track[n-1][1][0]
                y1 = self.track[n-1][1][1]
                t1 = self.track[n-1][0]
            # Update direction of object
            self.compassDirection, self.direction = self.__direction_lookup(x1,x0,y1,y0)
            # Inside reference boundaries considering traffic direction:
            if self.__checkBoundary(x0,y0) and self.__checkBoundary(x1,y1):
                if TrafficObject.trafficUpdown:
                    distance = abs(np.polyval(TrafficObject.pff, y0) - np.polyval(TrafficObject.pff, y1)) # interpolated distance in meters
                else:
                    distance = abs(x0 - x1) * TrafficObject.xMeterPixel
                timeTravelled = (t1 - t0) / 1000 # time in seconds
                speed = (distance / timeTravelled) # speed in m/s
                return speed
            else:
                return self.speed # use last possible calculation
        else:
            return self.speed

    # Public method to get speed
    def getSpeed(self):
        return self.speed

    # Get Vehicle direction data
    def  __getVehicleDirection(self):
        return self.compassDirection, self.direction # Verbal and angle, video frame up is north 0°

    # Get vehicle class data
    def getVehicleClass(self):
        histogram = Counter(self.labels)
        percentHistogram = [(i, histogram[i] / len(self.labels)) for i in histogram]
        vClass, vClConf = percentHistogram[0]
        return vClass, vClConf, 0.0

    # Get vehicle color
    def __getVehicleColor(self):
        return "undefined"

    # Create MQTT message
    def __createMsg(self):
        vDirection,vAngle = self.__getVehicleDirection()
        vClass, vClassAvg, vClassStdDev = self.getVehicleClass()
        vColor = self.__getVehicleColor()

        return TrafficObject.jsonvehicleDataMsg.format(TrafficObject.capabilityAlternateId, TrafficObject.sensorAlternateId,
            self.id, vClass, self.__getSpeed(True), vAngle, vDirection, len(self.labels),
            len(self.track), vClassAvg, vClassStdDev, vColor)
