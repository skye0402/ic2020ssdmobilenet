import cv2
import numpy as np
import PySimpleGUI as sg
import argparse
import configparser
from os.path import basename 

mouse_pressed = False
windowName ="Define region of interest (ROI)"
windowRoi ="Selected ROI"

# Variables for the config file
referenceLength = 0.0
roiPolygon = []
upDown = True # Left/Right = False
imageSize = 0 # Width (leftRight) or Height (upDown)
configFile = "./config/measurements.ini"

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True        
        if len(roiPolygon)<4:
            print(x,y)
            cv2.circle(image_to_show, (x,y), 3, (0, 255, 0))
            roiPolygon.append((x,y))
        else:
            print("Press V to preview, press R to reset.")
        
# Main program
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="filename of video for detection")
ap.add_argument("-f", "--format", default="", help="Video '<width>' format to be used for display and tracker and output")
ap.add_argument("-cam", "--camera", action="store_true", help="Use camera as input")
args = vars(ap.parse_args())

fileName = basename(args["video"]) #Get Filename which will be the key later in the config file

config = configparser.ConfigParser()
config.read(configFile) #Read existing configuration

# Load video
if args["camera"]:
    vs = cv2.VideoCapture(0)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH,int(args["format"]))
else:
    vs = cv2.VideoCapture(args["video"])

ok, frame = vs.read()
nativeHeight, nativeWidth, _ = frame.shape

# Calculate new format if needed
if args["format"] != "":
    ratio = nativeHeight/nativeWidth
    newHeight = round(ratio*float(args["format"]))
    newWidth = int(args["format"])
    frame = cv2.resize(frame,(newWidth, newHeight)) #Bi-linear interpolation as default
else:
    newWidth = nativeWidth
    newHeight = nativeHeight

image_to_show = np.copy(frame)
cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, mouse_callback, frame)


while True:
    cv2.imshow(windowName, image_to_show)
    k = cv2.waitKey(1)

    if k == ord("s"): 
        # Save measurements
        floatRoi = []
        for pointXY in regionOfInterest:
            floatRoi.append((float(pointXY[0]/newWidth), float(pointXY[1]/newHeight)))

        try: #check if filename section exists
            config.set(fileName, "roi", str(floatRoi))
        except Exception as excpt:#NoSectionError:
            config.add_section(fileName)
            config.set(fileName, "roi", str(floatRoi))

        # Get the frame for the detector to maximize the network resolution
        minX = float(min(regionOfInterest, key=lambda item:item[0])[0] / newWidth)
        minY = float(min(regionOfInterest, key=lambda item:item[1])[1] / newHeight)
        maxX = float(max(regionOfInterest, key=lambda item:item[0])[0] / newWidth)
        maxY = float(max(regionOfInterest, key=lambda item:item[1])[1] / newHeight)

        config.set(fileName, "detectorFrame", str([(minX, minY),(maxX, maxY)]))
        with open(configFile, 'w') as configfile:
            config.write(configfile)
        break
    elif k == ord("v"):
        # Preview the ROI
        mask = np.zeros((newHeight,newWidth), dtype=np.uint8)
        regionOfInterest = roiPolygon
        cv2.fillPoly(mask, [np.array(regionOfInterest)], 1)
        image_to_show = np.copy(frame)
        newImage = cv2.bitwise_and(image_to_show,image_to_show, mask=mask)
        #cv2.copyTo(image_to_show, mask)q
        cv2.imshow(windowRoi, newImage)
    elif k == ord("r"):
        # Reset the points
        cv2.destroyWindow(windowRoi)
        roiPolygon = []
        image_to_show = np.copy(frame)
        cv2.imshow(windowName, image_to_show)
        print("Reset the points, please start over.")
    elif (k == 27) or (k == ord("q")):
        break


cv2.destroyAllWindows()        