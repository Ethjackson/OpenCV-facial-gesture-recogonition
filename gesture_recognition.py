def gesturerecognition():
    import numpy as np
    import cv2 as cv
    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    video = cv.VideoCapture(0) # begins video capture
    # create mp hands with its set parameters, set to 0.8 which while more lag is more then default 0.5 is more accurate with tracking and detection of hand
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
            # will go to else if camera not accessible
            if video.isOpened():
                while True:
                    ret, image = video.read() # captures frame by frame and stores in image variable, if frame read succesfully ret = true.

                    if not ret: # if ret = false close
                        print("frame not received. exiting now...")
                        break

                    image.flags.writeable = False
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # sets the image as rgb color 
                    image = cv.blur(image, (3,3))
                    results = hands.process(image) # processes the frame to the mediapipe hands

                    image.flags.writeable = True
                    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        image_height, image_width, c = image.shape # get image/video shape/dimensions
                        for hand_landmarks in results.multi_hand_landmarks:
                            # if there is hand landmarks on the screen, it will go through all landmarks shown and draw them onto the image with the above for loop and below mp_drawing
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            
                            # we can get points using mp_hands, if all fingers are bent to certain degree (finger tips lower then finger dips e.g half way up finger) except midle finger go into if statement 
                            if (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
                            and hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                            and hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
                            and hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y):
                                print("Profanity detected: middle finger")
                                # gets the start x & y and end x & y for middle finger being shown with the start being tip and end being base of finger and * by image height/width to find finger position relative to video
                                startx = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
                                starty = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
                                endx = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                                endy = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height) 
                                padding = 140 # padding is used to increase size of rectangle to cover profanity
                                # checks if start is less then end due to different - and + needed to cover finger from different angles
                                if startx > endx:
                                    cv.rectangle(image, (startx + padding, starty + padding), (endx - padding, endy - padding), (0, 0, 0), -1) 
                                else: 
                                    cv.rectangle(image, (startx - padding, starty - padding), (endx + padding, endy + padding), (0, 0, 0), -1)

                    # flip the image horizontally for a selfie display.
                    cv.imshow('Profanity Camera', cv.flip(image, 1))

                    # press q to close
                    if cv.waitKey(1) == ord('q'):
                        break
            else:
                print("Camera cannot open")
                exit() 

            video.release()
            cv.destroyAllWindows()

print("Profanity recognition starting, Camera will launch soon...")
gesturerecognition()