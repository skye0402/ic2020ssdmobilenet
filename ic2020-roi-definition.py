import cv2
import numpy as np
from imutils.video import FileVideoStream
import PySimpleGUI as sg
import argparse
import configparser
from os.path import basename 

mouse_pressed = False

# Variables for the config file
referenceLength = 0.0
referenceInPixels = []
upDown = True # Left/Right = False
imageSize = 0 # Width (leftRight) or Height (upDown)
configFile = "./config/measurements.ini"

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(image_to_show, (x,y), 3, (0, 255, 0))
        print(x,y)
        referenceInPixels.append((x,y))
        
# Main program
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="filename of video for detection")
ap.add_argument("-f", "--format", default="", help="Video '<width>' format to be used for display and tracker and output")
args = vars(ap.parse_args())

fileName = basename(args["video"]) #Get Filename which will be the key later in the config file

config = configparser.ConfigParser()
config.read(configFile) #Read existing configuration

# # construct the gui
# layout = [  [sg.Text("This tool is to create a measurement reference. You need to mark at least 2 identical lengths.")],
#             [sg.Text("What will the reference length be in meter?"), sg.InputText(key="-LENGTH-")],
#             [sg.Button("Ok"), sg.Button("Cancel")] ]

# # Create the Window

# window = sg.Window("IC2020 - Measurement tool for roads to determine speed", layout)
# while True:
#     event, values = window.read()
#     if event == sg.WIN_CLOSED or event == "Cancel":	# if user closes window or clicks cancel
#         print("Leaving program.")
#         quit()
#     if values["-LENGTH-"].isnumeric():
#         if float(values["-LENGTH-"]) > 0:

#             print("You entered ", values["-LENGTH-"],"m.")
#             referenceLength = float(values["-LENGTH-"])
#             window.close()
#             break
#     else:
#         window['-LENGTH-']('')

# Load video
vs = FileVideoStream(args["video"]).start()
frame = vs.read()
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
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback, frame)


while True:
    cv2.imshow("image", image_to_show)
    k = cv2.waitKey(1)

    if k == ord("s"): 
        # Save measurements
        try: #check if filename section exists
            config.set(fileName, "format", str((newWidth, newHeight)))
            break
        except Exception as excpt:#NoSectionError:
            config.add_section(fileName)
            config.set(fileName, "format", str((newWidth, newHeight)))
        config.set(fileName, "trafficUpDown", str(upDown))
        config.set(fileName, "referenceLenght", str(referenceLength))
        config.set(fileName, "references",str(referenceInPixels))
        with open(configFile, 'w') as configfile:
            config.write(configfile)
        break
    elif k == 27:
        break

cv2.destroyAllWindows()        