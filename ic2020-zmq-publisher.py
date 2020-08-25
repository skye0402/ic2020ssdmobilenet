# import the necessary packages
import imagezmq
import cv2
import argparse
import socket
import time

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server-ip", required=True, help="ip address of the server to which the client will connect")
    ap.add_argument("-v", "--video", required=True, help="path to video file to be streamed")
    ap.add_argument("-f", "--format", help="format of the streamed video if it should be resized (width)")
    args = vars(ap.parse_args())

     # Publish on port
    port = 5555
    sender = imagezmq.ImageSender("tcp://*:{}".format(port), REQ_REP=False)

    # get the host name, initialize the video stream
    publisherName = socket.gethostname()
    vs = cv2.VideoCapture(args["video"])
    ok, frame = vs.read()
    if ok:
        nativeHeight, nativeWidth, _ = frame.shape
        # Calculate new format if needed
        if args["format"] != "":
            ratio = nativeHeight/nativeWidth
            newHeight = round(ratio*float(args["format"]))
            newWidth = int(args["format"])
        else:
            newWidth = nativeWidth
            newHeight = nativeHeight
        
        # JPEG quality, 0 - 100
        jpeg_quality = 95
        counter = 0
    try:
        while True:
            # read the frame from the camera and send it to the server
            ok, frame = vs.read()
            if not ok: break
            frame = cv2.resize(frame,(newWidth, newHeight))
            ret_code, jpg_buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            sender.send_jpg(publisherName, jpg_buffer)
            # print("Sent frame {}".format(counter))
            # counter = counter + 1
    except (KeyboardInterrupt, SystemExit):
        print('Exit due to keyboard interrupt')
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()
    finally:
        capture.stop()
        sender.close()
        sys.exit()