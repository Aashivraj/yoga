import cv2
import numpy as np
import websockets
import asyncio
import mediapipe as mp
import json
import os


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

json_file_path = 'json/squats.json'


if not os.path.exists('json'):
    os.makedirs('json')
with open(json_file_path, 'w') as json_file:
    json.dump([], json_file, indent=4)



landmark_names = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE",
    "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY",
    "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
    "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


frame_count = 0 
frame_skip = 5

async def handle_video_feed(websocket, path):
    global frame_count
    try:
        await websocket.send("Welcome to the server!")
        print("Client connected for real-time video feed.")

        
        async for message in websocket:
            if isinstance(message, bytes):
                
                np_arr = np.frombuffer(message, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
 
                # Convert the frame to RGB (for MediaPipe)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # reduce resolution
                frame = cv2.resize(frame, (640, 480))
                rotated_frame = cv2.rotate(frame , cv2.ROTATE_90_CLOCKWISE)

                # Process the frame with MediaPipe Pose detection
                results = pose.process(image_rgb)

                # If landmarks are detected, draw them on the frame and save to JSON
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Collect landmark data for this frame
                    frame_landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        frame_landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })

                    
                    
                    frame_data = {
                        'landmarks': frame_landmarks
                    }
                    
                    # Continuously append the current frame's landmarks to the JSON file
                    with open(json_file_path, 'r+') as json_file:
                        # Read the current JSON data
                        data = json.load(json_file)
                        # Append new landmarks
                        data.append(frame_data)
                        # Move file pointer to the beginning and overwrite
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)

                # Display the processed frame with landmarks
                cv2.imshow("Live Video Feed with Pose Detection", rotated_frame)

                # Close the window if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except Exception as e:
        print(f"Video feed error: {e}")
    finally:
        print("Client disconnected from video feed.")
        cv2.destroyAllWindows()

# Start the WebSocket server for video feed
async def start_video_feed_server():
    async with websockets.serve(handle_video_feed, "0.0.0.0", 5001, ping_interval= 40, ping_timeout=60):
        print("WebSocket server for video feed is running on ws://0.0.0.0:5001/")
        await asyncio.Future()

# Run the WebSocket server
if __name__ == '__main__':
    asyncio.run(start_video_feed_server())
