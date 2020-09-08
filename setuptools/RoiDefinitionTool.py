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

# Variables for the config file
referenceLength = 0.0
mode = "D" #Define
roiPolygon = []
regionOfInterest = []
roiAmount = 1
detectorBox = []
dSquare = []
upDown = True # Left/Right = False
imageSize = 0 # Width (leftRight) or Height (upDown)
configFile = "./config/measurements.ini"

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed
    if mode = "D":
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
            # elif len(roiPolygon)>=4*roiAmount and len(detectorBox)<roiAmount: # from here we set the detector box upper, left corner
            #     colorCode = (255,255,255)
            #     print("Detector box {:d} upper left corner ".format(len(detectorBox)),x,y)
            #     cv2.circle(image_to_show, (x,y), 3, color=colorCode, thickness=-1)
            #     detectorBox.append((x,y))
        
            else:
                print("Press V to preview, press R to reset.")
    elif mode = "S":
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(image_to_show,(),(), 1, colorCode)
        
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

# # construct the gui
# layout = [  [sg.Text("Please define the ")],
#             [sg.Text("e.g. 2.0 = tracking window is 2x detection ROI etc."), sg.InputText(key="-TRACKROI-")],
#             [sg.Button("Ok"), sg.Button("Cancel")] ]

# # Create the Window

# window = sg.Window("ROI tool for roads to define tracking area relative to detection ROI", layout, finalize=True)
# while True:
#     event, values = window.read()
#     if event == sg.WIN_CLOSED or event == "Cancel":	# if user closes window or clicks cancel
#         print("Leaving program.")
#         quit()
#     if values["-TRACKROI-"].isnumeric():
#         if float(values["-TRACKROI-"]) >= 1.0:

#             print("You entered that tracking window will be ", values["-TRACKROI-"]," times the detection ROI.")
#             trackingRoi = float(values["-TRACKROI-"])
#             window.close()
#             break
#     else:
#         window['-TRACKROI-']('')


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

    if (k == ord("s") and len(roiPolygon)==4*roiAmount): 
        # Save measurements
        floatRoi = []
        for pointXY in roiPolygon:
            floatRoi.append((float(pointXY[0]/newWidth), float(pointXY[1]/newHeight)))

        try: #check if filename section exists
            config.set(fileName, "roi", str(floatRoi))
        except Exception as excpt:#NoSectionError:
            config.add_section(fileName)
            config.set(fileName, "roi", str(floatRoi))

        # # Get the frame for the detector to maximize the network resolution
        # minX = float(min(regionOfInterest, key=lambda item:item[0])[0] / newWidth)
        # minY = float(min(regionOfInterest, key=lambda item:item[1])[1] / newHeight)
        # maxX = float(max(regionOfInterest, key=lambda item:item[0])[0] / newWidth)
        # maxY = float(max(regionOfInterest, key=lambda item:item[1])[1] / newHeight)

        config.set(fileName, "detectorFrame", str([(minX, minY),(maxX, maxY)]))
        with open(configFile, 'w') as configfile:
            config.write(configfile)
        break
    elif k == ord("v") and len(roiPolygon)==4*roiAmount:
        # Preview the ROI
        mode = "S" # Set square
        mask = np.zeros((newHeight,newWidth), dtype=np.uint8)
        
        for i in range(1,roiAmount+1):
            regionOfInterest = roiPolygon[:4*i]
            cv2.fillPoly(mask, [np.array(regionOfInterest)], 1)
            # Build detector frame square

            minX = min(regionOfInterest, key=lambda item:item[0])[0]
            minY = min(regionOfInterest, key=lambda item:item[0])[1]
            maxX = max(regionOfInterest, key=lambda item:item[0])[0]
            maxY = max(regionOfInterest, key=lambda item:item[0])[1]
            dX = maxX - minX
            dY = maxY - minY
            dSquare.append(min(dX, dY)) #Square edge length (pixels)

        image_to_show = np.copy(frame)
        image_to_show = cv2.bitwise_and(image_to_show,image_to_show, mask=mask)
        cv2.imshow(windowName, image_to_show)
        # Now show squares

    elif k == ord("r"):
        # Reset the points
        cv2.destroyWindow(windowName)
        roiPolygon = []
        dSquare = []
        mode = "D"
        image_to_show = np.copy(frame)
        cv2.imshow(windowName, image_to_show)
        cv2.setMouseCallback(windowName, mouse_callback, frame)
        print("Reset the points, please start over.")
    elif (k == 27) or (k == ord("q")):
        break


cv2.destroyAllWindows()        