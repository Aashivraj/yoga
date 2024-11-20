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
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

# Load correct form data
try:
    with open('json/5 2.json', 'r') as json_file:
        correct_form = json.load(json_file)
except FileNotFoundError:
    logger.error("JSON file not found")
    correct_form = []

threshold = 0.4

# Ensure templates directory exists
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
os.makedirs(template_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def process_frame(frame):
    """Process a single frame with pose detection and return feedback"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    feedback_data = {
        'has_pose': False,
        'feedback_text': '',
        'joint_errors': [],
        'overall_error': 0
    }
    
    if results.pose_landmarks:
        feedback_data['has_pose'] = True
        
        # Extract landmarks
        live_frame_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            live_frame_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        
        # Calculate errors
        joint_errors = []
        for i in range(len(live_frame_landmarks)):
            error = np.sqrt(np.sum((np.array([live_frame_landmarks[i]['x'], 
                                            live_frame_landmarks[i]['y'], 
                                            live_frame_landmarks[i]['z']]) -
                                   np.array([correct_form[0][i]['x'],
                                           correct_form[0][i]['y'],
                                           correct_form[0][i]['z']])) ** 2))
            joint_errors.append(float(error))
        
        feedback_data['joint_errors'] = joint_errors
        feedback_data['overall_error'] = float(np.mean(joint_errors))
        feedback_data['feedback_text'] = "Pose matches!" if feedback_data['overall_error'] < threshold else "Adjust your pose!"
        
        # Draw pose on frame
        height, width = frame.shape[:2]
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if joint_errors[start_idx] > threshold or joint_errors[end_idx] > threshold:
                color = (0, 0, 255)  # Red for errors
            else:
                color = (0, 255, 0)  # Green for correct
                
            start_point = (int(results.pose_landmarks.landmark[start_idx].x * width),
                          int(results.pose_landmarks.landmark[start_idx].y * height))
            end_point = (int(results.pose_landmarks.landmark[end_idx].x * width),
                        int(results.pose_landmarks.landmark[end_idx].y * height))
            
            cv2.line(frame, start_point, end_point, color, 2)
    
    return frame, feedback_data

async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    logger.info("New client connected")
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    # Decode base64 image
                    img_data = base64.b64decode(msg.data.split(',')[1])
                    np_arr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    # Process frame
                    processed_frame, feedback_data = process_frame(frame)
                    
                    # Encode processed frame
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send back both image and feedback data
                    response = {
                        'image': f"data:image/jpeg;base64,{processed_frame_b64}",
                        'feedback': feedback_data
                    }
                    
                    await ws.send_json(response)
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    await ws.send_json({'error': str(e)})
                    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("Client disconnected")
        return ws

async def init_app():
    # Create aiohttp app
    aio_app = web.Application()
    
    # Add routes
    aio_app.router.add_route('GET', '/ws', handle_websocket)
    
    # Add Flask app using WSGIHandler
    wsgi_handler = WSGIHandler(app)
    aio_app.router.add_route('*', '/{path_info:.*}', wsgi_handler.handle_request)
    
    return aio_app

if __name__ == '__main__':
    web.run_app(init_app(), host='0.0.0.0', port=5000)