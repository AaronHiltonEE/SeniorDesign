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
# ***PID START*** (IMPORT GPIO CONTROL)
import RPi.GPIO	as GPIO
# ***PID END***

# START COPY
from imutils.video import VideoStream
import numpy as np
# END COPY

# ***PID START***
'''
initialize gpio pins,
start gpio,
k-PID
'''
# initialize motor control pins
global IN1
IN1 = 2
global IN2
IN2 = 3
global ENA
INA = 4
global IN3
IN3 = 17
global IN4
IN4 = 27
global ENB
ENB = 22

# gain for tilt
global kpt
kpt = float(0.01)
global kit
kit = float(0.01)
global kdt
kdt = float(0)

# gain for pano
global kpp
kpp = float(0.01)
global kip
kip = float(0.01)
global kdp
kdp = float(0)

# desired pixel loc for drone.
global desired_tilt
desired_tilt = 250
global desired_pan
desired_pan = 250

# start all motor control processes.
GPIO.setmode(GPIO.BCM)

# initialize pin/output
GPIO.setup(IN1, GPIO.OUT)
GPIO.output(IN1, GPIO.LOW)

GPIO.setup(IN2, GPIO.OUT)
GPIO.output(IN2, GPIO.LOW)

GPIO.setup(IN3, GPIO.OUT)
GPIO.output(IN3, GPIO.LOW)

GPIO.setup(IN4, GPIO.OUT)
GPIO.output(IN4, GPIO.LOW)

GPIO.setup(ENA, GPIO.OUT)
GPIO.output(ENB, GPIO.OUT)

# start pwm control
tilt = GPIO.PWM(ENA, GPIO.OUT)
tilt.start(0)

pan = GPIO.PWM(ENB, GPIO.OUT)
pan.start(0)


# ***PID END***

class PhotoBoothApp:
	global net
	global PERSON
	
	# ***PID START***
	
	# ***PID END***
	
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
		#***START PID***
		# Initialize both counter for differential and integrator - pad and tilt
		rate_error_tilt = 0
		sum_error_tilt = 0
		error_tilt_old = 0
		rate_error_pan = 0
		sum_error_pan = 0
		error_pan_old = 0	
		pid_tilt = 0
		pid_pan = 0
		turret_tilt = 85
		turret_pan = 130 
		#***END PID***
		
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
				
				# ***START PID***
				drone_tilt_location = ycord
				drone_pan_location = xcord

				# PROPORTIONAL control variable Tilt
				error_tilt = kpt*(desired_tilt - drone_tilt_location)
				# PROPORTIONAL control variable Pano
				error_pan = kpp*(desired_pan - drone_pan_location)

				# INTEGRAL control variable Tilt
				sum_error_tilt = sum_error_tilt + error_tilt
				# INTRGRAL control variable Pano
				sum_error_pan = sum_error_pan + error_pan

				# DIFFERENCE control variable Tilt
				rate_error_tilt = error_tilt - error_tilt_old
				error_tilt_old = error_tilt
				# DIFFERENCE control variable Pano
				rate_error_pan = error_pan - error_pan_old
				error_pan_old = error_pan

				# Calculation and Logic for PID return
				pid_tilt = kpt*(error_tilt) + kit*(sum_error_tilt) + kdt*(rate_error_tilt)
				pid_pan = kpp*(error_pan) + kip*(sum_error_pan) + kdp*(rate_error_pan)

				if pid_pan > 100:
					pid_pan = 100
				elif pid_pan < -100:
					pid_pan = -100
				else:
					pid_pan = pid_pan
				if pid_tilt > 100:
					pid_tilt = 100
				elif pid_tilt < -100:
					pid_tilt = -100
				else:
					pid_tilt = pid_tilt

				if pid_tilt == 0:
					Tilt_Stop()
				elif pid_tilt>0:
					Tilt_Up(pid_tilt)
				elif pid_tilt<0:
					Tilt_Down(pid_tilt)
				elif pid_pan==0:
					Pan_Stop()
				elif pid_pan>0:
					Pan_Right(pid_pan)
				elif pid_pan<0:
					Pan_Left(pid_pan)
				else:
					Tilt_Stop()
					Pan_Stop()		

				def Tilt_Stop():
					GPIO.output(IN1,GPIO.LOW)
					GPIO.output(IN2,GPIO.LOW)

				def Pan_Stop():
					GPIO.output(IN1,GPIO.LOW)
					GPIO.output(IN2,GPIO.LOW)

				def Tilt_Up(pid_tilt):
					GPIO.output(IN1,GPIO.HIGH)
					GPIO.output(IN2,GPIO.LOW)
					t.ChangeDutyCycle(abs(pid_tilt))

				def Tilt_Down(pid_tilt):
					GPIO.output(IN1,GPIO.LOW)
					GPIO.output(IN2,GPIO.HIGH)
					t.ChangeDutyCycle(abs(pid_tilt))

				def Pan_Right(pid_pan):
					GPIO.output(IN1,GPIO.HIGH)
					GPIO.output(IN2,GPIO.LOW)
					p.ChangeDutyCycle(abs(pid_pan))

				def Pan_Left(pid_pan):
					GPIO.output(IN1,GPIO.LOW)
					GPIO.output(IN2,GPIO.HIGH)
					p.ChangeDutyCycle(abs(pid_pan))
				
				# ***PID END***
				
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
	

	
