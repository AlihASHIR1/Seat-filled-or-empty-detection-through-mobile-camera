# Seat-filled-or-empty-detection-through-mobile-camera
It is a chair / seat filled or empty detection in a specific area through connection of code with the mobile camera



This repository contains a code designed to detect and monitor seat occupancy in real-time using a mobile device's camera. By leveraging mediapipe framework, this code identifies whether a seat is "Occupied" or "Vacant," making it ideal for managing social distancing, library seating, or public transport capacity.

## Some basic requirments for the code:
1. You need to download some useful libraries and frameworks to be used:
   * cv2, os, urllib, mediapipe
2. You need to download the iVCam software in both mobile and PC.
3. You need to connect both the devices to the same network to connect both with each other.

In this code **"efficientdet lite0.tflite model"** has been used which is a pretrained model and is directly imported using url.

The code runs a mathematical check for every single chair on screen. It asks: "Is there any 'Person Box' overlapping with this 'Chair Box' and than provides us the result that the chair is occupied or not.
