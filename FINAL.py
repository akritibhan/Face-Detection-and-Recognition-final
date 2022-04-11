import cv2 as cv
import os
import numpy as np
import face_recognition as fr
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
from datetime import datetime


def face_Detection():
    upload1()
    capture = cv.VideoCapture(0)
    cascade_classifier = cv.CascadeClassifier(
        'Face_Detection/haarcascade_frontalface_default.xml')

    while True:
        isTrue, frame = capture.read()

        detections = cascade_classifier.detectMultiScale(frame, 1.3, 5)

        if (len(detections) > 0):
            (x, y, w, h) = detections[0]
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow('Face Detection', frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


def face_Recognition():
    upload2()

    path = 'Face_Recognition/images'
    images = []
    names = []
    myList = os.listdir(path)

    for i in myList:
        currentImg = cv.imread(f'{path}/{i}')
        images.append(currentImg)
        names.append(os.path.splitext(i)[0])

    def encodings(images):
        encoding_list = []
        for i in images:
            i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
            encode = fr.face_encodings(i)[0]
            encoding_list.append(encode)
        return encoding_list

    encoding_list_known = encodings(images)

    capture = cv.VideoCapture(0)

    while True:
        isTrue, frame = capture.read()

        frame_small = cv.resize(frame, (0, 0), None, 0.25, 0.25)
        frame_small = cv.cvtColor(frame_small, cv.COLOR_BGR2RGB)

        face_loc_frame = fr.face_locations(frame_small)
        encode_frame = fr.face_encodings(frame_small, face_loc_frame)

        for encodeFace, locFace in zip(encode_frame, face_loc_frame):
            matches = fr.compare_faces(encoding_list_known, encodeFace)
            faceDis = fr.face_distance(encoding_list_known, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = names[matchIndex].upper()
                y1, x2, y2, x1 = locFace
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.rectangle(frame, (x1, y2+37), (x2, y2), (0, 255, 0), -1)
                cv.putText(frame, name, (x1+6, y2+30),
                           cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Face Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


def mark_Attendance():
    upload3()

    path = 'Mark_Attendance/images'
    images = []
    names = []
    myList = os.listdir(path)

    for i in myList:
        currentImg = cv.imread(f'{path}/{i}')
        images.append(currentImg)
        names.append(os.path.splitext(i)[0])

    def markAttendance(name):
        with open('Attendance_Sheet.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

    def encodings(images):
        encoding_list = []
        for i in images:
            i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
            encode = fr.face_encodings(i)[0]
            encoding_list.append(encode)
        return encoding_list

    encoding_list_known = encodings(images)

    capture = cv.VideoCapture(0)

    while True:
        isTrue, frame = capture.read()

        frame_small = cv.resize(frame, (0, 0), None, 0.25, 0.25)
        frame_small = cv.cvtColor(frame_small, cv.COLOR_BGR2RGB)

        face_loc_frame = fr.face_locations(frame_small)
        encode_frame = fr.face_encodings(frame_small, face_loc_frame)

        for encodeFace, locFace in zip(encode_frame, face_loc_frame):
            matches = fr.compare_faces(encoding_list_known, encodeFace)
            faceDis = fr.face_distance(encoding_list_known, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = names[matchIndex].upper()
                y1, x2, y2, x1 = locFace
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.rectangle(frame, (x1, y2+37), (x2, y2), (0, 255, 0), -1)
                cv.putText(frame, name, (x1+6, y2+30),
                           cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv.imshow('Mark Attendance', frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


def object_Detection():
    upload4()

    config_file = 'Object_Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'Object_Detection/frozen_inference_graph.pb'

    model = cv.dnn_DetectionModel(frozen_model, config_file)

    classLabels = []
    file_name = 'Object_Detection/Labels.txt'
    with open(file_name, 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')

    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    capture = cv.VideoCapture('Object_Detection/Object_Detection_Video.mp4')

    def rescaleFrame(frame, scale=0.25):
        width= int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)

        dimensions = (width, height)

        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    while True:
        isTrue, frame = capture.read()

        frame_resized = rescaleFrame(frame, scale=0.25)
        
        ClassIndex, confidece, bbox = model.detect(frame_resized, confThreshold=0.55)

        if (len(ClassIndex) != 0):
            for ClassInd, conf, boxes, in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
                if(ClassInd <= 80):
                    cv.rectangle(frame_resized, boxes, (255, 0, 0), 2)
                    cv.putText(frame_resized, classLabels[ClassInd-1], (boxes[0] + 10,
                                                                boxes[1] + 30), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

        cv.imshow('Object Detection', frame_resized)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


def open_Excel_Sheet():
    os.startfile('Attendance_Sheet.csv')


def upload1():
    import time
    statusvar1.set("Loading.")
    sbar1.update()
    time.sleep(0.5)
    statusvar1.set("Loading..")
    sbar1.update()
    time.sleep(0.5)
    statusvar1.set("Loading...")
    sbar1.update()
    time.sleep(0.5)
    statusvar1.set("Hold On!")
    sbar1.update()
    statusvar1.set("")


def upload2():
    import time
    statusvar2.set("Loading.")
    sbar2.update()
    time.sleep(0.5)
    statusvar2.set("Loading..")
    sbar2.update()
    time.sleep(0.5)
    statusvar2.set("Loading...")
    sbar2.update()
    time.sleep(0.5)
    statusvar2.set("Hold On!")
    sbar2.update()
    statusvar2.set("")


def upload3():
    import time
    statusvar3.set("Loading.")
    sbar3.update()
    time.sleep(0.5)
    statusvar3.set("Loading..")
    sbar3.update()
    time.sleep(0.5)
    statusvar3.set("Loading...")
    sbar3.update()
    time.sleep(0.5)
    statusvar3.set("Hold On!")
    sbar3.update()
    statusvar3.set("")


def upload4():
    import time
    statusvar4.set("Loading.")
    sbar4.update()
    time.sleep(0.5)
    statusvar4.set("Loading..")
    sbar4.update()
    time.sleep(0.5)
    statusvar4.set("Loading...")
    sbar4.update()
    time.sleep(0.5)
    statusvar4.set("Hold On!")
    sbar4.update()
    statusvar4.set("")


root = Tk()
root.geometry("1000x600")
root.maxsize(1000, 600)
root.minsize(1000, 600)
root.title("Face Recognition And Detecting Features")
root.configure(bg="#8646f1")

heading1 = Label(root, text="FACE RECOGNITION & DETECTION FEATURES",
                 bg="#8646f1", fg="#ededed", font="tahoma 30 bold")
heading1.pack(side=TOP, pady=(40, 5))
heading2 = Label(root, text="Made By: Akriti & Anurag",
                 bg="#8646f1", fg="#ededed", font="tahoma 20 bold")
heading2.pack(side=TOP, pady=(5, 0))

frame1 = Frame(root, bg="#ededed", padx=5, pady=5)
name1 = Label(frame1, fg="#722ce1", bg="#ededed",
              text="FACE DETECTION", font="lucida 12 bold")
name1.pack(pady=(15, 10))
text1 = Label(frame1, width=25, height=8, fg="black", bg="#ededed",
              text="This will help you to detect\nfaces in live video supported\nby webcam.", font="verdana 10")
text1.pack(pady=10)
statusvar1 = StringVar()
statusvar1.set("")
sbar1 = Label(frame1, textvariable=statusvar1, bg="#ededed",
              fg="#722ce1", font="tahoma 10 bold", pady=5)
sbar1.pack(side=BOTTOM, fill=X)
button1 = Button(frame1, width=16, bg="#8846f1", fg="#ededed", text="Click to Enter",
                 font="lucida 15 bold", relief=GROOVE, command=face_Detection)
button1.pack(side=BOTTOM)
frame1.pack(side=LEFT, padx=20)

frame2 = Frame(root, bg="#ededed", padx=5, pady=5)
name2 = Label(frame2, fg="#722ce1", bg="#ededed",
              text="FACE RECOGNITION", font="lucida 12 bold")
name2.pack(pady=(15, 10))
text2 = Label(frame2, width=25, height=8, fg="black", bg="#ededed",
              text="This will help you to recognise\nfaces in live video supported\nby webcam.", font="verdana 10")
text2.pack(pady=10)
statusvar2 = StringVar()
statusvar2.set("")
sbar2 = Label(frame2, textvariable=statusvar2, bg="#ededed",
              fg="#722ce1", font="tahoma 10 bold", pady=5)
sbar2.pack(side=BOTTOM, fill=X)
button2 = Button(frame2, width=16, bg="#8846f1", fg="white", text="Click to Enter",
                 font="lucida 15 bold", relief=GROOVE, command=face_Recognition)
button2.pack(side=BOTTOM)
frame2.pack(side=LEFT, padx=20)

frame3 = Frame(root, bg="#ededed", padx=5, pady=5)
name3 = Label(frame3, fg="#722ce1", bg="#ededed",
              text="ATTENDANCE SYSTEM", font="lucida 12 bold")
name3.pack(pady=(15, 10))
text3 = Label(frame3, width=22, height=5, fg="black", bg="#ededed",
              text="This will help you recognise\nface of a person and mark\nattendance of that person\n in an excel sheet.", font="verdana 10")
text3.pack(pady=10)
statusvar3 = StringVar()
statusvar3.set("")
sbar3 = Label(frame3, textvariable=statusvar3, bg="#ededed",
              fg="#722ce1", font="tahoma 10 bold", pady=5)
sbar3.pack(side=BOTTOM, fill=X)
button3 = Button(frame3, width=16, bg="#8846f1", fg="white", text="Click to Enter",
                 font="lucida 15 bold", relief=GROOVE, command=mark_Attendance)
button3.pack(side=BOTTOM)
button_extra = Button(frame3, width=16, bg="#8846f1", fg="white", text="Open Excel Sheet",
                      font="lucida 15 bold", relief=GROOVE, command=open_Excel_Sheet)
button_extra.pack(side=BOTTOM, pady=5)
frame3.pack(side=LEFT, padx=20)

frame4 = Frame(root, bg="#ededed", padx=5, pady=5)
name3 = Label(frame4, fg="#722ce1", bg="#ededed",
              text="OBJECT DETECTION", font="lucida 12 bold")
name3.pack(pady=(15, 10))
text4 = Label(frame4, width=22, height=8, fg="black", bg="#ededed",
              text="This will detect objects\nfrom a video.", font="verdana 10")
text4.pack(pady=10)
statusvar4 = StringVar()
statusvar4.set("")
sbar4 = Label(frame4, textvariable=statusvar4, bg="#ededed",
              fg="#722ce1", font="tahoma 10 bold", pady=5)
sbar4.pack(side=BOTTOM, fill=X)
button4 = Button(frame4, width=16, bg="#8846f1", fg="white", text="Click to Enter",
                 font="lucida 15 bold", relief=GROOVE, command=object_Detection)
button4.pack(side=BOTTOM)
frame4.pack(side=LEFT, padx=20)

root.mainloop()
