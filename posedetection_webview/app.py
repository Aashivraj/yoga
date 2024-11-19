from flask import Flask, render_template
import asyncio
import cv2
import numpy as np
import base64
from flask_cors import CORS
import logging
from aiohttp import web
import aiohttp
from aiohttp_wsgi import WSGIHandler
import mediapipe as mp
import json

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load correct form from JSON
with open('json/5 2.json', 'r') as json_file:
    correct_form = json.load(json_file)

# Constants
threshold = 0.4
highlight_threshold = 0.25

def calculate_joint_errors(live_frame, correct_frame):
    joint_errors = []
    for i in range(len(live_frame)):
        error = np.sqrt(np.sum((np.array([live_frame[i]['x'], live_frame[i]['y'], live_frame[i]['z']]) -
                               np.array([correct_frame[i]['x'], correct_frame[i]['y'], correct_frame[i]['z']])) ** 2))
        joint_errors.append(error)
    return joint_errors

def draw_skeleton_with_feedback(frame, landmarks, joint_errors, threshold):
    height, width, _ = frame.shape
    faulty_landmarks = []
    connections = mp_pose.POSE_CONNECTIONS
    pixel_landmarks = [(int(l.x * width), int(l.y * height)) for l in landmarks]

    for idx, joint_error in enumerate(joint_errors):
        if joint_error > threshold:
            faulty_landmarks.append(idx)

    for connection in connections:
        start_idx, end_idx = connection
        start_point = pixel_landmarks[start_idx]
        end_point = pixel_landmarks[end_idx]

        color = (0, 0, 255) if start_idx in faulty_landmarks or end_idx in faulty_landmarks else (0, 255, 0)
        cv2.line(frame, start_point, end_point, color, 2)

    for idx, (cx, cy) in enumerate(pixel_landmarks):
        color = (0, 255, 0) if idx not in faulty_landmarks else (0, 0, 255)
        cv2.circle(frame, (cx, cy), 3, color, -1)

    return frame

def process_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        live_frame_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            live_frame_landmarks.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
        
        for correct_frame in correct_form:
            joint_errors = calculate_joint_errors(live_frame_landmarks, correct_frame)
            frame = draw_skeleton_with_feedback(frame, results.pose_landmarks.landmark, joint_errors, threshold)
            
            overall_error = np.mean(joint_errors)
            feedback_text = "Pose matches!" if overall_error < threshold else f"Adjust your pose! Error: {overall_error:.4f}"
            cv2.putText(frame, feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return frame, overall_error if 'overall_error' in locals() else None

@app.route('/')
def index():
    return render_template('index.html')

async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    logger.info("New client connected")
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    # Convert base64 string to image
                    img_data = base64.b64decode(msg.data.split(',')[1])
                    np_arr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    # Process frame with pose detection
                    processed_frame, error_value = process_frame(frame)
                    
                    # Convert processed frame back to base64
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Create response with processed frame and error value
                    response = {
                        'frame': f'data:image/jpeg;base64,{processed_frame_base64}',
                        'error': float(error_value) if error_value is not None else None
                    }
                    
                    # Send response back to client
                    await ws.send_str(json.dumps(response))
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    await ws.send_str(json.dumps({'error': str(e)}))
                    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("Client disconnected")
        return ws

async def init_app():
    aio_app = web.Application()
    aio_app.router.add_route('GET', '/ws', handle_websocket)
    wsgi_handler = WSGIHandler(app)
    aio_app.router.add_route('*', '/{path_info:.*}', wsgi_handler.handle_request)
    return aio_app

if __name__ == '__main__':
    web.run_app(init_app(), host='0.0.0.0', port=5000)