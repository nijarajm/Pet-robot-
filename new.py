from __future__ import print_function
import cv2 as cv
import argparse
import telepot
import pyttsx3
import time

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [create]

def rescaleFrame(frame,scale=1):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
## [capture]
capture =   cv.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]


while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)

    if frame_resized is None:
        
        break


    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame_resized)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame_resized, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame_resized, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ret,thresh= cv.threshold(fgMask,1,255,0)

     # Find the contours of the foreground objects
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if cv.countNonZero(thresh) > 0:
 
        # Convert the frame to grayscale
            gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    
            # Apply adaptive thresholding using a Gaussian window
            thres = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

            cnts, hierarchy = cv.findContours(thres, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

         
            for contour in cnts :
                    area = cv.contourArea(contour)
                    line = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
                    perimeter = cv.arcLength(contour, True)
                    approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)
                    if area >=300:
                       if abs(line[1]/line[0]) < 0.1:
                            continue
                       elif len(approx) > 4:
                            continue
                       else:
                          cv.drawContours(frame_resized,[contour],0,(255,0,0),2)
                          x, y, w, h = cv.boundingRect(contour)
                          cv.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                          break
                       
    #voice = pyttsx3.init()
    #voice.say('water detected') 
     
    bot = telepot.Bot('6274286363:AAHaTXt0r_ftm01FC5uAAi_CiYucxUGdne0')
  #Send a message to your Telegram bot
    bot.sendMessage(chat_id='1008149071', text='/detected')
    #time.sleep(5)
    
                    
    cv.imshow('Frame', frame_resized)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('thresh',thresh)
    #voice.runAndWait()
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()


# Close all windows
cv.destroyAllWindows()