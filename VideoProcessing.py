# This handles video input from file, streaming over network or attached camera

import cv2
from time import sleep
import threading
import socket
import imagezmq
import subprocess


class VideoProcessor:
    # commands to adjust Camera parameters by video4linux2 control tool
    autoExpCommand = "v4l2-ctl -d0 --set-ctrl exposure_auto=3"
    manualExpCommand = "v4l2-ctl -d0 --set-ctrl exposure_auto=1"
    manualExpValueCommand = "v4l2-ctl -d0 --set-ctrl exposure_absolute=" # 50 to 10000 (default 166)
    gainCommand = "v4l2-ctl -d0 --set-ctrl gain=" # gain, 1-128 (default 64)
    gammaCommand = "v4l2-ctl -d0 --set-ctrl gamma=" #gamma, 100-500 (default 300) 

    shellFlag = True

    def __init__(self, hostname, port, video, camera, format):
        self.videoInput = ""
        self.hostname = hostname
        self.port = port
        self.video = video
        self.message = ""
        # default settings if not specified else
        self.gamma = 300
        self.gain = 64
        self.exposure = 166
        self.autoExposure = 3 # 3: automatic, 1: manual setting

        if camera:
            self.videoInput = "Camera"
            #self.vs = cv2.VideoCapture(0)
            #self.vs.release()
            self.vs = cv2.VideoCapture(0, cv2.CAP_V4L)
            self.vs.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)           
            sleep(2) #wait 2 seconds for sensor to boot up
        elif hostname != "" and hostname != None:
            self.videoInput == "Streaming"
            self.receiver = VideoStreamSubscriber(hostname, port) # Create subscriber instance            
        else:
            # Load video file
            self.videoInput = "File"
            self.vs = cv2.VideoCapture(args["video"])

    def getFrame(self):
        if self.videoInput == "Streaming":
            self.message, frame = self.receiver.receive()
            ok = True # TODO: Is there a way to get a success message?
            return ok, cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)
        else:
            ok, frame = self.vs.read()
            return ok, frame

    def getTimestamp(self):
        if self.videoInput == "Streaming":
            return time()*1000
        else:
            return self.vs.get(cv2.CAP_PROP_POS_MSEC)

    def close(self):
        if self.videoInput == "Streaming":
            self.receiver.close()
        else:
            self.vs.release()

    def getCameraParams(self):
        # Returns the current parameter values
        if self.videoInput == "Camera":
            if self.autoExposure == 3:
                exposureMethod = "Auto"
            else:
                exposureMethod = "Manual"
            return exposureMethod, self.exposure, self.gain, self.gamma
        else:
            return "No camera source", 0, 0, 0
    
    def setCameraParams(self, inputKey):
        if inputKey ==ord("I"): # Controls gain (no impact on exposure timing)
            if self.gain<128: 
                self.gain += 2
                output = subprocess.call(self.gainCommand+str(self.gain), shell=self.shellFlag)
        elif inputKey ==ord("i"):
            if self.gain>1: 
                self.gain -= 1
                output = subprocess.call(self.gainCommand+str(self.gain), shell=self.shellFlag)
        elif inputKey ==ord("G"): #Gamma higher
            if self.gamma<500: 
                self.gamma += 50
                output = subprocess.call(self.gammaCommand+str(self.gamma), shell=self.shellFlag)
        elif inputKey ==ord("g"): #Gamma lower
            if self.gamma>100: 
                self.gamma -= 50
                output = subprocess.call(self.gammaCommand+str(self.gamma), shell=self.shellFlag)
        elif inputKey ==ord("A") or inputKey ==ord("a"): #Auto-Exposure
            if self.autoExposure == 1: 
                output = subprocess.call(self.autoExpCommand, shell=self.shellFlag)
                self.autoExposure = 3
            else: 
                output = subprocess.call(self.manualExpCommand, shell=self.shellFlag)
                self.autoExposure = 1
        elif inputKey ==ord("X"): # Controls gain (no impact on exposure timing)
            if self.exposure<10000: 
                self.exposure += 20
                output = subprocess.call(self.manualExpValueCommand+str(self.exposure), shell=self.shellFlag)
        elif inputKey ==ord("x"):
            if self.exposure>50: 
                self.exposure -= 20
                output = subprocess.call(self.manualExpValueCommand+str(self.exposure), shell=self.shellFlag)
        elif inputKey ==ord("P") or inputKey ==ord("p"): #Program controlled
            #TODO: Logic missing
            sleep(1)

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