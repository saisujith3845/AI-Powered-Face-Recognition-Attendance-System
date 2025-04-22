import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time

# Function to take image of the user
def TakeImage(l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    # Check for empty fields and alphanumeric name validation
    if (l1 == "") and (l2 == ""):
        t = 'Please enter your Enrollment Number and Name.'
        text_to_speech(t)
    elif l1 == '':
        t = 'Please enter your Enrollment Number.'
        text_to_speech(t)
    elif l2 == "":
        t = 'Please enter your Name.'
        text_to_speech(t)
    elif not l2.isalnum():
        t = 'Name must be alphanumeric. Please enter a valid name.'
        text_to_speech(t)
    else:
        try:
            # Initialize camera and face detector
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(haarcasecade_path)
            Enrollment = l1
            Name = l2
            sampleNum = 0
            directory = Enrollment + "_" + Name
            path = os.path.join(trainimage_path, directory)
            os.mkdir(path)

            # Capture 80 face images for training
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    cv2.imwrite(
                        f"{path}/{Name}_{Enrollment}_{sampleNum}.jpg",
                        gray[y:y + h, x:x + w],
                    )
                    cv2.imshow("Frame", img)

                # Exit conditions for capturing images
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elif sampleNum >= 80:
                    break

            # Release camera and close window
            cam.release()
            cv2.destroyAllWindows()

            # Write user details to CSV file
            row = [Enrollment, Name]
            with open("StudentDetails/studentdetails.csv", "a+", newline="") as csvFile:
                writer = csv.writer(csvFile, delimiter=",")
                writer.writerow(row)
            
            res = "Images saved for ER No: " + Enrollment + " Name: " + Name
            message.configure(text=res)
            text_to_speech(res)

        except FileExistsError:
            F = "Student data already exists."
            text_to_speech(F)
