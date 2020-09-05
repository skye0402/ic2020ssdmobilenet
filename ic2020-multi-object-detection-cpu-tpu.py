# Multi-object tracking for traffic Team 304 // Innovator Challenge 2020

# import the necessary modules
from edgetpu.detection.engine import DetectionEngine
import tflite_runtime.interpreter as tflite
import imutils
from collections import Counter
import cv2
from PIL import Image
import argparse
from time import time
from time import sleep
from uuid import uuid4
import platform
import ssl
import paho.mqtt.client as mqtt
import configparser
from sklearn.linear_model import LinearRegression
import numpy as np
from os.path import basename 
import threading
import socket
import imagezmq

def make_interpreter(tpu, model_file):
    if tpu:
        model_file, *device = model_file.split('@')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
        ])
    else:
        return tflite.Interpreter(model_file)

def getInterpreterDetails(interpreter):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    return input_details, output_details, input_shape

def invokeInterpreter(interpreter, inputdetails, outputdetails, image):
    # Call tensorflow lite
    interpreter.set_tensor(inputdetails, [image])
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    return boxes, classes, scores

def getBoundingBox(boxData, newWidth, newHeight, detectorImageRectangle):
    (startY, startX, endY, endX) = boxData
    # Calculate the detection boxes based on the detector image
    x0,y0 = detectorImageRectangle[0]
    x1,y1 = detectorImageRectangle[1]

    detectorWidth = x1-x0
    detectorHeight = y1-y0

    refX = newWidth - x0
    refY = newHeight - y0

    startX = 0 if startX < 0 else int(startX * detectorWidth) + x0
    startY = 0 if startY < 0 else int(startY * detectorWidth - ((detectorWidth - detectorHeight)/2)) + y0
    endX = 0 if endX < 0 else int(endX * detectorWidth) + x0
    endY = 0 if endY < 0 else int(endY * detectorWidth - ((detectorWidth - detectorHeight)/2)) + y0
    return (startX, startY, endX, endY)

def resizeAndPadImage(image, maskImg, networkSize, detectorImageRectangle):
    # 1. Apply mask to image
    maskedImage = cv2.bitwise_and(image,image, mask=maskImg)
    # cv2.imshow("ROI", maskedImage)
    # k = cv2.waitKey(0)

    # 2. Cut detector rectangle out of image
    x0,y0 = detectorImageRectangle[0]
    x1,y1 = detectorImageRectangle[1]
    detectorImage = maskedImage[y0:y1,x0:x1]
    # cv2.imshow("ROI", detectorImage)
    # k = cv2.waitKey(0)

    # 3. Create the padded, masked, sub-image for the detector network
    detectorSize = detectorImage.shape[:2]
    ratio = float(networkSize)/max(detectorSize)
    new_size = tuple([int(x*ratio) for x in detectorSize])

    # new_size should be in (width, height) format
    image = cv2.resize(detectorImage, (new_size[1], new_size[0]))

    delta_w = networkSize - new_size[1]
    delta_h = networkSize - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# The callback for when the client receives a CONNACK response from the server.
def onConnect(client, userdata, flags, rc):
    rcList = {
        0: "Connection successful",
        1: "Connection refused - incorrect protocol version",
        2: "Connection refused - invalid client identifier",
        3: "Connection refused - server unavailable",
        4: "Connection refused - bad username or password",
        5: "Connection refused",
    }
    print(rcList.get(rc, "Unknown server connection return code {}.".format(rc)))

# The callback for when a PUBLISH message is received from the server.
def onMessage(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

# Send message to SAP MQTT Server
def sendMessage(client, deviceID, messageContentJson):
    client.publish(deviceID, messageContentJson)    

def startMqttClient(deviceId):
    client = mqtt.Client(deviceId) 
    client.on_connect = onConnect
    client.on_message = onMessage
    client.tls_set(certfile=pemCertFilePath, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS, ciphers=None)
    client.connect(mqttServerUrl, mqttServerPort)
    client.subscribe(ackTopicLevel+sapIotDeviceID) #Subscribe to device ack topic (feedback given from SAP IoT MQTT Server)
    client.loop_start() #Listening loop start
    return client

class VideoProcessor:
    def __init__(self, hostname, port, video, camera, format):
        self.videoStream = False
        self.hostname = hostname
        self.port = port
        self.video = video
        self.message = ""

        if camera:
            self.videoStream = False
            #self.vs = cv2.VideoCapture(0)
            #self.vs.release()
            self.vs = cv2.VideoCapture(0, cv2.CAP_V4L)
            self.vs.set(cv2.CAP_PROP_FRAME_WIDTH,1280)#int(format))
            self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            #self.vs.set(cv2.CAP_PROP_FPS, 30)
            
            sleep(2) #wait 2 seconds for sensor to boot up
        elif hostname != "" and hostname != None:
            self.videoStream = True
            self.receiver = VideoStreamSubscriber(hostname, port) # Create subscriber instance            
        else:
            # Load video file
            self.vs = cv2.VideoCapture(args["video"])

    def getFrame(self):
        if self.videoStream:
            self.message, frame = self.receiver.receive()
            ok = True # TODO: Is there a way to get a success message?
            return ok, cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)
        else:
            ok, frame = self.vs.read()
            return ok, frame

    def getTimestamp(self):
        if self.videoStream:
            return time()*1000
        else:
            return self.vs.get(cv2.CAP_PROP_POS_MSEC)

    def close(self):
        if self.videoStream:
            self.receiver.close()
        else:
            self.vs.release()

# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        while not self._stop:
            self._data = receiver.recv_jpg()
            self._data_ready.set()
        receiver.close()

    def close(self):
        self._stop = True

# this class holds the detected objects and tracks them
class TrafficObject:
    # Class variables
    imgHeight = 0
    imgWidth = 0
    pff = [] # Fitting factors (delivered by main program)
    minPixel = 0 # Earliest/ latest measurement points
    maxPixel = 0 
    framesForSpeedCalc = 0 # last n frames to consider for speed calculation
    maxXBoundary = 0 #Coordinate where tracker should stop
    maxYBoundary = 0 #set by main program
    labels = [] # Labels for the classes
    allowedClasses = [] # Classes to be considered

    # Constructor method
    def __init__(self, tTracker, bBox, classLabel, creditLimit, detectionMissDebit, maxCredit, timestamp):
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

    # Destructor method
    def __del__(self):
        jsonMsg = self.__createMsg()
        sendMessage(mqttClient, measuresTopicLevel+sapIotDeviceID, jsonMsg) # send the object data to SAP IoT
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
        return labels[classID]

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
        if (cX >= TrafficObject.maxXBoundary) or (cY >= TrafficObject.maxYBoundary) or (cX <= 0) or (cY <= 0):
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

    # Calculate speed
    def __getSpeed(self, useAllData=False, nFrames = 2):
        speed = 0
        n = len(self.track)
        if n>=nFrames: # Check we have sufficient measurements
            if useAllData: # We measure from first trackpoint to current trackpoint (higher precision?)
                for yP in self.track:
                    y0 = yP[1][1] 
                    t0 = yP[0]
                    if y0 >= TrafficObject.minPixel and y0 <= TrafficObject.maxPixel: break
                for yP in reversed(self.track):
                    y1 = yP[1][1] 
                    t1 = yP[0]
                    if y1 >= TrafficObject.minPixel and y1 <= TrafficObject.maxPixel: break
            else:
                y0 = self.track[n-nFrames][1][1]        
                t0 = self.track[n-nFrames][0]
                y1 = self.track[n-1][1][1]
                t1 = self.track[n-1][0]
            # Inside reference boundaries:
            if y0>=TrafficObject.minPixel and y0<=TrafficObject.maxPixel and y1>=TrafficObject.minPixel and y1<=TrafficObject.maxPixel: 
                distance = abs(np.polyval(TrafficObject.pff, y0) - np.polyval(TrafficObject.pff, y1)) # interpolated distance in meters
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
        return "Inbound", 179.5 # Verbal and angle, video frame up is north 0°

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

        return jsonvehicleDataMsg.format(capabilityAlternateId, sensorAlternateId,
            self.id, vClass, self.__getSpeed(True), vAngle, vDirection, len(self.labels),
            len(self.track), vClassAvg, vClassStdDev, vColor)

if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(['./config/tracker.ini', "./config/mqtt.ini", "./config/measurements.ini"])

    # -------------- Parameters ------------------>>>
    # MQTT
    mqttServerUrl = config.get("server","mqttServerUrl")
    mqttServerPort = config.getint("server","mqttServerPort")
    pemCertFilePath = config.get("server","pemCertFilePath")
    sapIotDeviceID = config.get("devices","sapIotDeviceID")
    sensorAlternateId = config.get("sensors","sensorAlternateId")
    capabilityAlternateId = config.get("sensors","capabilityAlternateId")
    ackTopicLevel = config.get("topics","ackTopicLevel")
    measuresTopicLevel = config.get("topics","measuresTopicLevel")
    jsonvehicleDataMsg = config.get("messages", "vehicleData").replace("\n","")

     # Tracker
    trafficDict = {} # Contains the list of objects found
    trackerType = config.get("tracker","trackerType")
    ioUThreshold = config.getfloat("tracker","ioUThreshold")
    staleObject = config.getint("detector","staleObject")
    detectionCredit = config.getint("detector","detectionCredit")
    maxCredit = config.getint("detector","maxCredit")
    detectionMissDebit = config.getint("detector","detectionMissDebit")
    detectionConfidence = config.getfloat("detector","detectionConfidence")
    maxTrackerBoxSize = config.getfloat("tracker","maxTrackerBoxSize")
    classAnomalies = config.get("detector","classAnomalies")
    framesForSpeedCalc = config.getint("tracker","framesForSpeedCalc")
    allowedClasses = eval(config.get("classes","allowedClass"))
    # -------------- Parameters ------------------<<<

    EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
    }[platform.system()]

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to TensorFlow Lite object detection model")
    ap.add_argument("-l", "--labels", required=True, help="path to labels file")
    ap.add_argument("-c", "--confidence", type=float, default=0.65, help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--video", required=True, help="filename of video for detection")
    ap.add_argument("-t", "--tpu", action="store_true", help="TPU is present/ should be used")
    ap.add_argument("-f", "--format", default="", help="Video '<width>' format to be used for display and tracker and output")
    ap.add_argument("-hl", "--headless", action="store_true", help="Headless mode, no video output")
    ap.add_argument("-o", "--output", default="", help="write video to filename </file>")
    ap.add_argument("-hn", "--hostname", help="hostname of publisher")
    ap.add_argument("-pt", "--port", default=5555, help="port of publisher")
    ap.add_argument("-cam", "--camera", action="store_true", help="use USB connected camera")
    args = vars(ap.parse_args())

    #Get Filename which will be the key later in the config file
    fileName = basename(args["video"]) 
    # Measurements reference file for speed
    referenceLength = config.getfloat(fileName, "referencelenght")
    trafficupdown = config.getboolean(fileName, "trafficupdown")
    distanceReference = eval(config.get(fileName, "references"))
    regionOfInterest = eval(config.get(fileName, "roi"))
    detectorReference = eval(config.get(fileName, "detectorframe"))

    # start MQTT client
    mqttClient = startMqttClient(sapIotDeviceID)

    # initialize the labels dictionary
    print("[INFO] parsing class labels...")
    labels = {}

    # loop over the class labels file
    for row in open(args["labels"]):
        # unpack the row and update the labels dictionary
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()
    TrafficObject.labels = labels
    TrafficObject.allowedClasses = allowedClasses

    # load the tflite detection model
    print("[INFO] loading model into TF Lite...")
    interpreter = make_interpreter(args["tpu"], args["model"])
    input_details, output_details, net_input_shape = getInterpreterDetails(interpreter)
    interpreter.allocate_tensors()

    # Get required input size of frame
    networkSize = input_details[0]['shape'].max()

    # Prepare for the video
    print("[INFO] starting video stream...")
    trafficVideo = VideoProcessor(args["hostname"],args["port"],args["video"], args["camera"], args["format"])
    ok, orig = trafficVideo.getFrame()

    nativeHeight, nativeWidth, _ = orig.shape
    # Start the video processing
    # Calculate new format if needed
    if args["format"] != "":
        ratio = nativeHeight/nativeWidth
        newHeight = round(ratio*float(args["format"]))
        newWidth = int(args["format"])
    else:
        newWidth = nativeWidth
        newHeight = nativeHeight
    TrafficObject.imgHeight = newHeight # Set the format of the frame
    TrafficObject.imgWidth = newWidth

    # Calculate float data from config file to pixel data for chosen format
    pixelReference = []
    for pointXY in distanceReference:
        pixelReference.append((round(pointXY[0]*newWidth), round(pointXY[1]*newHeight)))

    detectorinPixel = []
    for pointXY in detectorReference:
        detectorinPixel.append((round(pointXY[0]*newWidth), round(pointXY[1]*newHeight)))

    roiInPixel = []
    for pointXY in regionOfInterest:
        roiInPixel.append((round(pointXY[0]*newWidth), round(pointXY[1]*newHeight)))

    # Create mask image from configuration file
    maskImage = np.zeros((newHeight,newWidth), dtype=np.uint8)
    cv2.fillPoly(maskImage, [np.array(roiInPixel)], 1)

    # Calculate fitting for distance measurement
    referenceDataList = np.asarray(pixelReference, dtype=np.float)
    pixelY = referenceDataList[:,1] # Get height reference pixels (Y-Axis)
    meterScale = np.linspace(0, referenceLength*(len(pixelReference)-1), len(pixelReference)) # Create meter-scale
    polyFitFactors = np.polyfit(pixelY, meterScale, 4) #Polynominal fitting of n-th grade
    TrafficObject.pff = polyFitFactors # Send to class variable
    TrafficObject.minPixel = min(pixelY)
    TrafficObject.maxPixel = max(pixelY)
    TrafficObject.framesForSpeedCalc = framesForSpeedCalc
    TrafficObject.maxXBoundary, TrafficObject.maxYBoundary = detectorinPixel[1] # Tracker ends here

    # Optional: write video out
    writeVideo = False
    if args["output"] != "":
        writeVideo = True
        fourcc = cv2.VideoWriter_fourcc(*'DIVX') #(*'MP42')
        out = cv2.VideoWriter(args["output"], fourcc, 20.0, (newWidth,newHeight))

    start = time()
    # loop over the frames from the video stream
    while ok:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of x pixels
        ok, orig = trafficVideo.getFrame()
        if not ok: #End of video reached
            break
        fpstimer = cv2.getTickCount()
        if args["format"] != "": orig = cv2.resize(orig,(newWidth, newHeight)) #Bi-linear interpolation as default
        frame = resizeAndPadImage(orig, maskImage, networkSize, detectorinPixel)
        # cv2.imshow("New detector image", frame)
        # cv2.waitKey(0)
        
        # from BGR to RGB channel ordering 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # carry out detections on the input frame
        boxes, classes, scores = invokeInterpreter(interpreter, input_details[0]['index'], output_details, frame)
        
        # Update active trackers
        timestamp = trafficVideo.getTimestamp()
        for objectID in list(trafficDict):
            objectOk = trafficDict[objectID].updateTracker(orig, timestamp)
            if not objectOk:
                del trafficDict[objectID] #remove the tracker

        # loop over the confidence results
        for index, conf in enumerate(scores):
            if not ((conf > detectionConfidence) and (conf <= 1.0) and (TrafficObject.isClassRelevant(int(classes[index])))): break
            # extract the bounding box and predicted class label
            box = getBoundingBox(boxes[index].flatten(), newWidth, newHeight, detectorinPixel)      
            label = TrafficObject.getClassName(classes[index].astype("int"))
            startX, startY, endX, endY = box
            if (float(endX-startX)/newWidth) > maxTrackerBoxSize: break # Skips unlikely big detections
            
            # Tracking handling starts here
            tBox = (startX, startY, endX-startX, endY-startY) # Detected box in tracker format
            objectFound = False
            for objectID in list(trafficDict):
                if trafficDict[objectID].getIoU(box) > ioUThreshold:  #IoU threshold is met?             
                    trafficDict[objectID].addLabel(label)
                    trafficDict[objectID].addCount(TrafficObject.createTracker(trackerType), orig, tBox, detectionCredit)
                    objectFound = True

            if objectFound == False: #No matching tracker, let's add a new tracker
                timestamp = trafficVideo.getTimestamp()
                tObject = TrafficObject(TrafficObject.createTracker(trackerType), box, label, staleObject, detectionMissDebit, maxCredit, timestamp)
                tObject.tracker.init(orig, tBox)
                trafficDict[tObject.getId] = tObject

        for objectID in list(trafficDict):
            if (trafficDict[objectID].getCredit() == 0) or (trafficDict[objectID].isGone()): #Remove stale or disappeared objects
                del trafficDict[objectID]
            else:
                startXY, endXY = trafficDict[objectID].getBBoxCoord()
                cv2.rectangle(orig, startXY, endXY, (0, 255, 0), 1)
                y = startXY[1] - 15 if startXY[1] - 15 > 15 else startXY[1] + 15
                vClass, _, _ = trafficDict[objectID].getVehicleClass() # Class of car
                vId = trafficDict[objectID].getId()[-4:] # Vehicle ID (last 4 digits)
                vSpeed = trafficDict[objectID].getSpeed()
                vCredit = trafficDict[objectID].getCredit() # Show internal trust into tracked object
                text = "{0!s} ID{1!s}: v={2:2.1f}km/h {3:d}% credit".format(vClass, vId , vSpeed*3.6, vCredit)
                cv2.putText(orig, text, (startXY[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Tracker handling ends here

        # print time required
        # print(f"Image resize: |{conv_time*1000}|ms. RGB conv.: |{rgb_time*1000}|ms. Inference: |{inf_time*1000}|ms.")
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - fpstimer)
        text = "fps: {:.0f}".format(fps)
        cv2.putText(orig, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # show the output frame and wait for a key press
        if not args["headless"]:
            cv2.imshow("Frame", orig)
        # else:
        #     print(text)
        key = cv2.waitKey(1) & 0xFF
        
        # Optional: write video
        if writeVideo: out.write(orig)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("Quit command received.")
            break

    # do a bit of cleanup
    overall_time = time() - start #Time to convert the frame
    print(overall_time)
    if writeVideo: out.release()
    cv2.destroyAllWindows()
    trafficVideo.close()
    mqttClient.loop_stop
    print("CV2 tasks and MQTT client stopped.")