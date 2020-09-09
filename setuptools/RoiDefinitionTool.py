import cv2
import numpy as np
import PySimpleGUI as sg
import argparse
import configparser
from time import sleep
from os.path import basename 

mouse_pressed = False
windowName ="Define region of interest (ROI)"
windowRoi ="Selected ROI"
maskedImage = []

# Variables for the config file
referenceLength = 0.0
mode = "D" #Define
roiPolygon = []
regionOfInterest = []
roiAmount = 1
detectorBox = []
detectorCount = 0
dRectangles = []
upDown = True # Left/Right = False
imageSize = 0 # Width (leftRight) or Height (upDown)
configFile = "./config/measurements.ini"
lastX = 0
lastY = 0

def mouse_callback(event, x, y, flags, param):
 
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed, detectorCount, dRectangles, maskedImage, lastX, lastY
    if mode == "D":
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pressed = True        
            if len(roiPolygon)>=4: # switch color for 2nd ROI polygon markers
                colorCode = (0,255,0)
            else:
                colorCode = (255,0,0)
            if len(roiPolygon)<4*roiAmount:
                print(x,y)
                cv2.circle(image_to_show, (x,y), 3, colorCode)
                roiPolygon.append((x,y))        
            else:
                print("Press V to preview, press R to reset.")

    elif mode == "S":
        # This code handles the detector windows. It starts with a square as this is ideal for using
        # the CNN (in current case 300x300px)
        # However, it might be benficial to make it rectangular using Xx and Yy keys (bigger, smaller)
        if detectorCount == 0:
            colorCode = (0,255,0) 
        else: 
            colorCode = (255,0,0)

        if event == cv2.EVENT_LBUTTONDOWN: # every left-click allows for replacing the detector box
            lastX = x
            lastY = y
            (x2, y2) = dRectangles[detectorCount]
            image_to_show = np.copy(maskedImage)
            if detectorCount>0:
                cv2.rectangle(image_to_show,detectorBox[0],detectorBox[1], color=(0,255,0))            
            cv2.rectangle(image_to_show,(x,y),(x+x2,y+y2), color=colorCode)

        if event == cv2.EVENT_RBUTTONDOWN: #Right-click saves the settings for the detector size and position
            (x2, y2) = dRectangles[detectorCount]
            detectorBox.append((lastX, lastY))
            detectorBox.append((lastX+x2, lastY+y2))
            print("Confirmed detector box ",detectorCount)
            detectorCount += 1
        
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

# construct the gui
layout = [  [sg.Text("This tool is to define the region(s) of interest (ROI). Each region is defined by a trapezoid.")],
            [sg.Text("You want one ROI per direction of traffic. How many will you define (1-2)?"), sg.InputText(key="-ROICOUNT-")],
            [sg.Button("Ok"), sg.Button("Cancel")] ]

# Create the Window

window = sg.Window("ROI tool for roads to define detection area", layout, finalize=True)
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Cancel":	# if user closes window or clicks cancel
        print("Leaving program.")
        quit()
    if values["-ROICOUNT-"].isnumeric():
        if int(values["-ROICOUNT-"]) > 0 and int(values["-ROICOUNT-"]) < 5:

            print("You entered ", values["-ROICOUNT-"]," ROIs.")
            roiAmount = int(values["-ROICOUNT-"])
            window.close()
            break
    else:
        window['-ROICOUNT-']('')

# Load video
if args["camera"]:
    vs = cv2.VideoCapture(0, cv2.CAP_V4L)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    vs.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
    sleep(4)
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
    if k<0: k=0
    k = chr(k)

    if (k == "s" and len(roiPolygon)==4*roiAmount): 
        # Save measurements
        floatRoi = []
        for pointXY in roiPolygon:
            floatRoi.append((float(pointXY[0]/newWidth), float(pointXY[1]/newHeight)))

        try: #check if filename section exists
            config.set(fileName, "roi", str(floatRoi))
        except Exception as excpt:#NoSectionError:
            config.add_section(fileName)
            config.set(fileName, "roi", str(floatRoi))

        detectorBoxFloat = []
        for xyC in detectorBox:
            relX = xyC[0] / newWidth
            relY = xyC[1] / newHeight
            detectorBoxFloat.append((relX,relY))

        config.set(fileName, "detectorFrame", str(detectorBoxFloat))
        with open(configFile, 'w') as configfile:
            config.write(configfile)
        break
    elif k == "v" and len(roiPolygon)==4*roiAmount:
        # Preview the ROI
        mode = "S" # Set square
        mask = np.zeros((newHeight,newWidth), dtype=np.uint8)
        
        for i in range(1,roiAmount+1):
            startIndex = 0
            if i > 1: startIndex = (i-1) * 4 - 1
            regionOfInterest = roiPolygon[startIndex:4*i]
            cv2.fillPoly(mask, [np.array(regionOfInterest)], 1)
            
            # Build detector frame square (starting point, might become a rectangle later)
            minX = min(regionOfInterest, key=lambda item:item[0])[0]
            minY = min(regionOfInterest, key=lambda item:item[1])[1]
            maxX = max(regionOfInterest, key=lambda item:item[0])[0]
            maxY = max(regionOfInterest, key=lambda item:item[1])[1]
            dX = int((maxX - minX)*0.8)
            dY = int((maxY - minY)*0.8)
            dRectangles.append((min(dX, dY),min(dX, dY))) #Square edge length (pixels)

        image_to_show = np.copy(frame)
        maskedImage = cv2.bitwise_and(image_to_show,image_to_show, mask=mask)
        image_to_show = np.copy(maskedImage)
        cv2.imshow(windowName, image_to_show)

    elif k == "r":
        # Reset the points
        cv2.destroyWindow(windowName)
        roiPolygon = []
        dRectangles = []
        detectorCount = 0
        mode = "D"
        image_to_show = np.copy(frame)
        cv2.imshow(windowName, image_to_show)
        cv2.setMouseCallback(windowName, mouse_callback, frame)
        print("Reset the points, please start over.")

    #Resizing of rectangle
    elif k in "xXyY":
        (xD, yD) = dRectangles[detectorCount]
        if k == "x":
            xD -= 1
        elif k == "X":
            xD += 1
        elif k == "y":
            yD -= 1
        elif k == "Y":
            yD += 1 
        dRectangles[detectorCount] = (xD, yD)

    elif (ord(k) == 27) or (k == "q"):
        break

cv2.destroyAllWindows()        