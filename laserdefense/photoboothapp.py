# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os

#Pulled from ...
# START COPY
from imutils.video import VideoStream
import numpy as np
# END COPY

class PhotoBoothApp:
	global net
	global PERSON
	def __init__(self, vs, outputPath, sm_prototxt, sm_model, sm_confidence, sm_movidius):
		
		#Pulled from real_time_object_detection.py...

		#START COPY
		# initialize the list of class labels MobileNet SSD was trained to
		# detect, then generate a set of bounding box colors for each class
		"""
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		"""
		CLASSES = ["person", "aeroplane"]
		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

		# load our serialized model from disk
		print("[INFO] loading model...")
		# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
		# Bypass the arguments and hardcord the 'prototxt' and 'model'.
		net = cv2.dnn.readNetFromCaffe(sm_prototxt, sm_model)
		# specify the target device as the Myriad processor on the NCS
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
		#END COPY
		
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.outputPath = outputPath
		self.frame = None
		self.thread = None
		self.stopEvent = None

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None

		# create two buttons shutdown and snapshot
		btn = tki.Button(self.root, bg = "#ce2029", pady = "15", text="Emergency Shutdown",
			command=self.onClose)
		btn.pack(side="bottom", fill="both", expand="yes", padx=10,
			pady=10)
		btn1 = tki.Button(self.root, text = "Snap Shot",
			command = self.takeSnapshot)
		btn1.pack(side = "bottom", fill = "both", expand = "yes", padx = 10,
			pady = 10)
		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=(net,COLORS))
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("LASER DEFENSE")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
		

	def videoLoop(self, net_pass, COLORS_pass):
		# DISCLAIMER:
		# Try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				#Pulled from real_time_object_detection.py...
				
				#START COPY
				
				# grab the frame from the video stream and resize it to
				# have a maximum width of 300 pixels
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=500)

				# grab the frame dimensions and convert it to a blob
				(h, w) = self.frame.shape[:2]
				blob = cv2.dnn.blobFromImage(self.frame, 0.007843, (500, 500), 127.5)

				# pass the blob through the network and obtain the detections and
				# predictions
				# net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
				net_pass.setInput(blob)
				detections = net_pass.forward()

				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated with
					# the prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections by ensuring the `confidence` is
					# greater than the minimum confidence
					if confidence > 0.78:
						# extract the index of the class label from the
						# `detections`, then compute the (x, y)-coordinates of
						# the bounding box for the object
						idx = int(detections[0, 0, i, 1])
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")

						# draw the prediction on the frame
						"""
						CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
						"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
						"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
						"sofa", "train", "tvmonitor"]
						"""
						CLASSES = ["person","aeroplane"]
						# hardcode CLASSES[0] instead of CLASSES[idx]
						label = "{}: {:.2f}%".format(CLASSES[0],
							confidence * 100)
						# hardcode CLASSES[0] instead of CLASSES[idx]	
						cv2.rectangle(self.frame, (startX, startY), (endX, endY),
							COLORS_pass[0], 2)
						y = startY - 15 if startY - 15 > 15 else startY + 15
						# hardcode CLASSES[0] instead of CLASSES[idx]
						cv2.putText(self.frame, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_pass[0], 2)
						
						# Bounding box with center dot.
						xcord = int((startX + endX)/2)
						ycord = int((startY + endY)/2)
						cv2.rectangle(self.frame, (startX,startY),(endX, endY), color= (0,0,255), thickness = 2)
						self.frame = cv2.circle(self.frame, (xcord,ycord), radius=2, color=(0,0,255),thickness = -5)

				# show the output frame
				# cv2.imshow("Frame", self.frame)
				key = cv2.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break
				#END COPY
				
				
				# grab the frame from the video stream and resize it to
				# have a maximum width of 300 pixels
				# self.frame = self.vs.read()
				# self.frame = imutils.resize(self.frame, width=300)
		
				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)
		
				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tki.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)
		
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image

		except RuntimeError as e:
			print("[INFO] caught a RuntimeError")

	def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.outputPath, filename))

		# save the file
		cv2.imwrite(p, self.frame.copy())
		print("[INFO] saved {}".format(filename))

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()
