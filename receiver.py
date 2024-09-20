import asyncio
import websockets
import cv2
import numpy as np
import base64

WEBSOCKET_SERVER = "ws://192.168.29.165:8765"

async def receive_frames(websocket):
    try:
        while True:
            data = await websocket.recv()
            print("Received data of size:", len(data))
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow('Received Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except websockets.ConnectionClosed:
        print("Connection to the WebSocket server closed.")
    finally:
        cv2.destroyAllWindows()

async def main():
    async with websockets.connect(WEBSOCKET_SERVER) as websocket:
        await receive_frames(websocket)

if __name__ == "__main__":
    asyncio.run(main())
