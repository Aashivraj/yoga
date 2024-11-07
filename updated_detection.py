import cv2
import numpy as np
import websockets
import asyncio
import mediapipe as mp
import json
import os


correct_form_file = 'json/fixed_video.json'
if not os.path.exists(correct_form_file):
    raise FileNotFoundError(f"Correct form JSON file '{correct_form_file}' not found.")
with open(correct_form_file, 'r') as file:
    correct_form_landmarks = json.load(file)




mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


threshold = 0.3
 



def calculate_joint_errors(live_frame, correct_frame):
    joint_errors = []
    for i in range(len(live_frame)):
        # Calculate Euclidean distance 
        error = np.sqrt(np.sum((np.array([live_frame[i]['x'], live_frame[i]['y'], live_frame[i]['z']]) -
                                np.array([correct_frame[i]['x'], correct_frame[i]['y'], correct_frame[i]['z']])) ** 2))
        joint_errors.append(error)
    return joint_errors




async def handle_video_feed(websocket, path):
    try:
        await websocket.send("Welcome to the server!")
        print("Client connected for real-time video feed.")

        frame_count = 0 

        async for message in websocket:
            if isinstance(message, bytes):
                
                # Convert message bytes to OpenCV frame
                np_arr = np.frombuffer(message, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                
                frame_count += 1

                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rotated_frame = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
                results = pose.process(rotated_frame)

                # print("==1==")
                # print(len(results.pose_landmarks.landmark) if results.pose_landmarks else 0)

                if results.pose_landmarks:
                    # Extract pose landmarks from the live frame
                    live_frame_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]

                    
                    
                    # Mirror the x-coordinates of all landmarks
                    # for lm in live_frame_landmarks:
                    #     lm['x'] = 1 - lm['x']  # Flip the x-axis by subtracting from 1
                        
                    # Select the correct reference frame from the JSON based on `frame_count`
                    correct_frame_index = min(frame_count, len(correct_form_landmarks) - 1)
                    correct_frame_landmarks = correct_form_landmarks[correct_frame_index]

                    matching_landmarks = []
                    
                    # print("==2==")
                    # print(len(matching_landmarks))
                    
                    
                    # Calculate joint errors for the current frame
                    joint_errors = calculate_joint_errors(live_frame_landmarks, correct_frame_landmarks)

                    
                    # print("==3==")
                    # print(len(matching_landmarks))
                    
                    # Prepare a list of dictionaries for each landmark's match status
                    for i, error in enumerate(joint_errors):
                        match_flag = error < threshold
                        matching_landmarks.append({
                            'x': live_frame_landmarks[i]['x'],
                            'y': live_frame_landmarks[i]['y'],
                            'z': live_frame_landmarks[i]['z'],
                            'landmark_index': i,  
                            'matches': bool(match_flag)  
                        })
                        
                        
                        
                        # print("==4==")
                        # print(len(matching_landmarks))

                    # Send the current frame's matching status as JSON data to the WebSocket client
                    frame_data = json.dumps(matching_landmarks)
                    
                    
                    
                    print("==5==")
                    # print(len(frame_data))
                    
                    print(frame_data)
                    
                    await websocket.send(frame_data)
                    print(f"Sent data for frame {frame_count}.")
                else:
                    await websocket.send("[]")
                    
                    
                    # print("==6==")
                    # print("[]")

    except Exception as e:
        print(f"Video feed error: {e}")
    finally:
        print("Client disconnected from video feed.")
        cv2.destroyAllWindows()


# Start the WebSocket server for video feed
async def start_video_feed_server():
    async with websockets.serve(handle_video_feed, "0.0.0.0", 5001, ping_interval=40, ping_timeout=60):
        print("WebSocket server for video feed is running on ws://0.0.0.0:5001/")
        await asyncio.Future()

# Run the WebSocket server
if __name__ == '__main__':
    asyncio.run(start_video_feed_server())