
''' Detect humans through image detection models in OpenCV '''

import cv2
import numpy as np
import imutils
import math

# haarcascades on GitHub
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

# local path to where OpenCV haarcascades are saved 
# ex. (Windows 10 Anocanda):  'C:\\Users\\bfurm\\Anaconda3\\pkgs\\libopencv-3.4.1-h875b8b8_3\\Library\\etc\\haarcascades\\'
LOCAL_PATH_TO_HAARCASCADES = 'resources/haarcascades/'

DELAY_BETWEEN_FRAMES_MILLISECONDS = 111  # 30 for smoothness on laptop, but it is a cpu workout
FRAMES_TO_LOCK_ONTO_FACE = 7

running = True


def start():
    global running
    running = True

    # get the models from OpenCV's files
    face_cascade = cv2.CascadeClassifier(LOCAL_PATH_TO_HAARCASCADES + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(LOCAL_PATH_TO_HAARCASCADES + 'haarcascade_eye.xml')

    # get the video from the main camera
    video_capture = cv2.VideoCapture(0)
	
    frames_in_a_row_with_face = 0

    # analyze frames coming in from video
    while running:
        
        # get the image and grayscale image from the video
        successful, image = video_capture.read()

        # image = cv2.imread('far.jpg')     #TEMP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # detect stuff with detectMultiScale(image, rejectLevels, levelweights?)
        faces = face_cascade.detectMultiScale(gray_image, 1.35, 4)  # faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        
        most_centered_face = None  # (x, y, w, h) tuple
        most_centered_face_dist_to_center = None
		
        if len(faces) > 0:
            frames_in_a_row_with_face += 1
        else:
            frames_in_a_row_with_face = 0
        
        
        for (x, y, w, h) in faces:
            if frames_in_a_row_with_face >= FRAMES_TO_LOCK_ONTO_FACE:  # different color when locked on
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 4)  #(x,y) is starting point, (x+w,y+h) is the ending point, (255,0,0) blue color, 2 is thickness
            else:
                cv2.rectangle(image, (x, y), (x+w, y+h), (122,0,0), 4) 
            
            # distance to center of image from center of face rectangle
            x_center, y_center = center(x,y,w,h)
            dist_to_center_screen = mag(compare_to_center(image, x_center, y_center))
            if most_centered_face is None or dist_to_center_screen < most_centered_face_dist_to_center:
                most_centered_face = (x, y, w, h)
                most_centered_face_dist_to_center = dist_to_center_screen
            
            '''
            # search for eyes inside the face: the region of interest
            roi_gray = gray_image[y:y+h,x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.25)  # eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            '''
            
        # draw the line to the center of the most centered face
        if most_centered_face is not None:
            x, y, w, h = most_centered_face
            x_center, y_center = center(x, y, w, h)
            xy_dist = compare_to_center(image, x_center, y_center, True)
            print('')
            print('face diagonal length: {},   direction to move: {}, {}'.format(mag((w,h)), xy_dist[0], xy_dist[1]))
        
        
        '''
        upper_body_cascade = cv2.CascadeClassifier(LOCAL_PATH_TO_HAARCASCADES + 'haarcascade_upperbody.xml')
        lower_body_cascade = cv2.CascadeClassifier(LOCAL_PATH_TO_HAARCASCADES + 'haarcascade_lowerbody.xml')
        full_body_cascade = cv2.CascadeClassifier(LOCAL_PATH_TO_HAARCASCADES + 'haarcascade_fullbody.xml')
        
        upper_bodies = upper_body_cascade.detectMultiScale(gray_image, 1.08, 2) # bodies = body_cascade.detectMultiScale(gray_image, 1.1, 4 )
        lower_bodies = lower_body_cascade.detectMultiScale(gray_image, 1.08, 2)
        full_bodies = full_body_cascade.detectMultiScale(gray_image, 1.08, 2)
        
        for (bx, by, bw, bh) in upper_bodies:
            cv2.rectangle(image, (bx,by), (bx+bw, by+bh), (0,0,255), 2)  #red rect for upper body
        for (bx, by, bw, bh) in lower_bodies:
            cv2.rectangle(image, (bx,by), (bx+bw, by+bh), (0,255,255), 2)  # yellow rect for lower body
        for (bx, by, bw, bh) in full_bodies:
            cv2.rectangle(image, (bx,by), (bx+bw, by+bh), (255,255,255), 2)  #white rect for full body
        '''
    
        # resize the window if it is too big
        image = imutils.resize(image, width=min(500, image.shape[1]))
            
        # show the image, then waits the DELAY_BETWEEN_FRAMES_MILLISECONDS
        cv2.imshow('Video',image)
        k = cv2.waitKey(DELAY_BETWEEN_FRAMES_MILLISECONDS) & 0xff
        if k == 27:  # Ends on the esc key
            stop()
            
    video_capture.release()
    cv2.destroyAllWindows()


def stop():
    global running
    running = False

	
# called when
def targetDetected():
	pass


# get the center of a rectangle
def center(x, y, w, h):
    return (x + w//2,  y + h//2)

# return the (x_dist, y_dist, face_width) from a point to the center of the image (left and up is negative)
def compare_to_center(image, x, y, is_drawing=False):   
    img_center_x = image.shape[1] // 2
    img_center_y = image.shape[0] // 2
    
    if is_drawing:
        # draw a line between the points of the center of the face, the center of the screen
        cv2.rectangle(image, (x, y), (x, y), (255,255,0), 7)
        cv2.rectangle(image, (img_center_x, img_center_y), (img_center_x, img_center_y), (255,255,0), 11)
        cv2.line(image, (x, y), (img_center_x, img_center_y), (111, 111, 111), 2)
        
    return (img_center_x - x, img_center_y - y)

# magnitude of a 2-length tuple
def mag(xy_tuple):
    if len(xy_tuple) != 2:
        return 0
    x, y = xy_tuple
    return math.sqrt(x**2 + y**2)




start()

 