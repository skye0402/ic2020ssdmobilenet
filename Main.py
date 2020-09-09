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
import platform
import configparser
from sklearn.linear_model import LinearRegression
import numpy as np
from os.path import basename 
import VideoProcessing
from Traffictracking import TrafficObject
import MqttClient

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

def calculateCentroid(vertexes):
    x_list = [vertex [0] for vertext in vertexes]
    y_list = [vertex [1] for vertext in vertexes]
    llen = len(vertexes)
    x = sum(x_list)/llen
    y = sum(y_list)/llen
    return (x,y)

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
    detectorAmount = int(len(detectorReference)/2) #Amount of detectors to be used

    # # Calculate boundary for tracking
    # for i in range(1, detectorAmount+1):
    #     roiCenter = calculateCentroid(regionOfInterest[i*4-4:i*4-1])
    #     detectorCenterX = (detectorReference[i*2-1]-detectorReference[i*2-2])/2
    #     detectorCenterY = (detectorReference[i*2]-detectorReference[i*2-1])/2


    # start MQTT client
    mqttClient = MqttClient.startMqttClient(sapIotDeviceID, pemCertFilePath, mqttServerUrl, mqttServerPort, ackTopicLevel, sapIotDeviceID)
    TrafficObject.sapIotDeviceID = sapIotDeviceID
    TrafficObject.mqttClient = mqttClient
    TrafficObject.measuresTopicLevel = measuresTopicLevel

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
    TrafficObject.jsonvehicleDataMsg = jsonvehicleDataMsg
    TrafficObject.capabilityAlternateId = capabilityAlternateId
    TrafficObject.sensorAlternateId = sensorAlternateId

    # load the tflite detection model
    print("[INFO] loading model into TF Lite...")
    interpreter = make_interpreter(args["tpu"], args["model"])
    input_details, output_details, net_input_shape = getInterpreterDetails(interpreter)
    interpreter.allocate_tensors()

    # Get required input size of frame
    networkSize = input_details[0]['shape'].max()

    # Prepare for the video
    print("[INFO] starting video stream...")
    trafficVideo = VideoProcessing.VideoProcessor(args["hostname"],args["port"],args["video"], args["camera"], args["format"])
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
    TrafficObject.roiArea = roiInPixel
    #TrafficObject.maxXBoundary, TrafficObject.maxYBoundary = detectorinPixel[1] # Tracker ends here

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
        # Loop for detectors
        detectorResults = []
        for i in range(1, detectorAmount+1):
            startIndex = i*2 - 2
            endIndex = i*2
            frame = resizeAndPadImage(orig, maskImage, networkSize, detectorinPixel[startIndex:endIndex])
            # cv2.imshow("New detector image", frame)
            # cv2.waitKey(0)
            
            # from BGR to RGB channel ordering 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # carry out detections on the input frame
            boxes, classes, scores = invokeInterpreter(interpreter, input_details[0]['index'], output_details, frame)
            detectorResults.append((boxes,classes,scores))
        
        # Update active trackers
        timestamp = trafficVideo.getTimestamp()
        for objectID in list(trafficDict):
            objectOk = trafficDict[objectID].updateTracker(orig, timestamp)
            if not objectOk:
                del trafficDict[objectID] #remove the tracker

        # loop over the confidence results
        for i in range(1, detectorAmount+1):
            boxes, classes, scores = detectorResults[i-1]
            sI = i*2-2
            eI = i*2
            currentDetectorinPixel =  detectorinPixel[sI:eI]  
    
            for index, conf in enumerate(scores):
                if not ((conf > detectionConfidence) and (conf <= 1.0) and (TrafficObject.isClassRelevant(int(classes[index])))): break
                # extract the bounding box and predicted class label
                box = getBoundingBox(boxes[index].flatten(), newWidth, newHeight, currentDetectorinPixel)      
                label = TrafficObject.getClassName(classes[index].astype("int"))
                startX, startY, endX, endY = box
                # Get center of detection box
                cX = startX + (endX-startX)/2
                cY = startY + (endY-startY)/2
                detectorWidth = currentDetectorinPixel[1][0]-currentDetectorinPixel[0][0]
                detectorWidthBorder = int((detectorWidth - detectorWidth * 0.8)/2) #TODO: Y and X depending on traffic flow and factor into config!

                if (cX<currentDetectorinPixel[0][0]+detectorWidthBorder) or (cX>currentDetectorinPixel[1][0]-detectorWidthBorder):
                    print("Outside of borders for ",i)
                    break
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
                    tObject = TrafficObject(TrafficObject.createTracker(trackerType), box, label, staleObject, detectionMissDebit, maxCredit, timestamp, i)
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
        autoex, exposure, gain, gamma = trafficVideo.getCameraParams()
        text = "fps: {:.0f} auto-exposure: {:s} exposure: {:d} gain: {:d} gamma: {:d}".format(fps, autoex, exposure, gain, gamma)
        cv2.putText(orig, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 

        # show the output frame and wait for a key press
        if not args["headless"]:
            cv2.imshow("Frame", orig)
        # else:
        #     print(text)
        key = cv2.waitKey(1) & 0xFF
        
        # Optional: write video
        if writeVideo: out.write(orig)

        # Allow for manual camera settings
        if chr(key) in "aAiIgGxXpP":
            trafficVideo.setCameraParams(key)

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