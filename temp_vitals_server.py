
import asyncio
import json
import random
import time
import websockets
from datetime import datetime

async def vitals_websocket_server(websocket, path):
    """Simulated vitals WebSocket server."""
    print(f"🔗 Vitals WebSocket connected: {websocket.remote_address}")
    
    try:
        while True:
            # Generate simulated vitals data
            vitals_data = {
                "timestamp": time.time(),
                "hr": random.randint(60, 100),  # Heart rate 60-100 BPM
                "spo2": random.randint(95, 100),  # SpO2 95-100%
                "quality": "good",
                "temperature": random.uniform(18, 30),  # Temperature 18-30°C
                "co2_level": random.randint(400, 1500)  # CO2 400-1500 PPM
            }
            
            # Send vitals data
            await websocket.send(json.dumps(vitals_data))
            print(f"📊 Sent vitals: HR={vitals_data['hr']}, SpO2={vitals_data['spo2']}, "
                  f"Temp={vitals_data['temperature']:.1f}°C, CO2={vitals_data['co2_level']} PPM")
            
            # Wait 1 second before next data
            await asyncio.sleep(1)
            
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Vitals WebSocket disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"❌ Vitals WebSocket error: {e}")

async def main():
    print("💓 Starting Vitals WebSocket Server...")
    server = await websockets.serve(vitals_websocket_server, "localhost", 8765)
    print("✅ Vitals WebSocket Server running on ws://localhost:8765")
    
    # Keep server running
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
