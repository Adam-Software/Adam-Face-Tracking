import time

#!/usr/bin/env python
import cv2

import numpy as np
import json
from typing import List
from Models.MotorCommand import MotorCommand
import websocket
import threading
from simple_pid import PID

FRAME_W = 1920 // 3
FRAME_H = 1080 // 3

WEBSOCKET_SERVER_URL = "ws://192.168.50.10:8000/adam-2.7/off-board"

class WebSocketClient:
    def __init__(self):
        self.ws = None
        self.is_connected = False
        self.lock = threading.Lock()

    def connect(self):
        with self.lock:
            if self.is_connected:
                return

            try:
                self.ws = websocket.WebSocketApp(
                    WEBSOCKET_SERVER_URL,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_close=self.on_close,
                    on_error=self.on_error,
                )
                self.is_connected = True
                thread = threading.Thread(target=self.ws.run_forever)
                thread.daemon = True
                thread.start()
            except Exception as e:
                print("Failed to connect:", e)

    def disconnect(self):
        with self.lock:
            if self.is_connected:
                self.ws.close()
                self.is_connected = False

    def send_data(self, data):
        with self.lock:
            if self.is_connected:
                data = json.dumps(data.__dict__, default=lambda x: x.__dict__)
                self.ws.send(data)

    def on_open(self, ws):
        print("WebSocket connection opened.")

    def on_message(self, ws, message):
        # Handle any incoming messages from the WebSocket server, if needed.
        pass

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed:", close_status_code, close_msg)
        self.is_connected = False

    def on_error(self, ws, error):
        print("WebSocket error:", error)
        self.is_connected = False

class SerializableCommands:
    motors: List[MotorCommand]

    def __init__(self, motors: List[MotorCommand]) -> None:
        self.motors = motors
# Generate a list of SerializableCommands objects
serializable_commands_list = []


def jsonCommandList(Head):
    command_list = [MotorCommand("head", Head[0]),
                    MotorCommand("neck", Head[1])]

    serializable_commands = SerializableCommands(command_list)

    return (serializable_commands)

class DataGenerator:
    def __init__(self):
        self.frame_count = 0

    def generate_data(self, HeadPercent):
        serializable_commands = jsonCommandList(HeadPercent)

        return serializable_commands


def frame_change_handler(HeadPercent):
    # Update frame count in the data dictionary
    data = data_generator.generate_data(HeadPercent)

    # Send data to the WebSocket server
    websocket_client.send_data(data)

# Create an instance of the WebSocketClient class
websocket_client = WebSocketClient()
data_generator = DataGenerator()

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def main():
    PIDW = PID(0.05, 0.005, 0.0025)
    PIDH = PID(0.07, 0.007, 0.0035)

    PIDW.setpoint = 0.15
    PIDH.setpoint = 0.15

    PIDW.output_limits = (-50, 50)
    PIDH.output_limits = (-50, 50)

    PIDW.sample_time = 0.00001
    PIDH.sample_time = 0.00001

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture("http://192.168.50.10:18000/stream/0.mjpeg")
    #cap = cv2.VideoCapture(0)

    no_move_zone_size = 5

    pan_percent = 50  # Initial pan (left-right) position percentage
    tilt_percent = 50  # Initial tilt (up-down) position percentage

    HeadPercent = [pan_percent, tilt_percent]
    frame_change_handler(HeadPercent)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            center_face = (x + w // 2, y + h // 2)
            center_image = (frame.shape[1] // 2, frame.shape[0] // 2)
            cv2.line(frame, center_image, center_face, (0, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)


            distance_to_center = ((center_face[0] - center_image[0])**2 + (center_face[1] - center_image[1])**2)**0.5
            cv2.rectangle(frame, (center_image[0] - no_move_zone_size, center_image[1] + no_move_zone_size),
                          (center_image[0] + no_move_zone_size, center_image[1] - no_move_zone_size), (0, 255, 255), 1)


            if distance_to_center <= no_move_zone_size:
                cv2.putText(frame, "No Move Zone", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:

                # Get the centre of the face
                x = x + (w / 2)
                y = y + (h / 2)

                # Correct relative to centre of image
                turn_x = float(x - (FRAME_W / 2))
                turn_y = float(y - (FRAME_H / 2))


                # Convert to percentage offset
                turn_x /= float(FRAME_W / 2)
                turn_y /= float(FRAME_H / 2)

                # Scale offset to degrees (that 2.5 value below acts like the Proportional factor in PID)
                turn_x *= 100  # VFOV
                turn_y *= 100  # HFOV

                controlX = PIDW(turn_x)
                controlY = PIDH(turn_y)

                pan_percent -=controlX
                tilt_percent += controlY

                print("Pan percent:", pan_percent,"Tilt percent:", tilt_percent,"\f")
                HeadPercent = [tilt_percent, pan_percent]
                frame_change_handler(HeadPercent)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    websocket_client.connect()
    main()
