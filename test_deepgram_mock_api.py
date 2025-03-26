#!/usr/bin/env python3
"""
Test script for the Deepgram Mock API. This script tests both REST and WebSocket endpoints.
"""

import os
import sys
import time
import json
import asyncio
import argparse
import requests
import websockets
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

# Default configuration
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_API_KEY = "default-api-key"

def test_health(api_url, api_key):
    """Test the health check endpoint."""
    print("\n=== Testing Health Check Endpoint ===")
    
    try:
        response = requests.get(f"{api_url}/health")
        
        if response.status_code == 200:
            print("Health check successful!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Health check failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing health check: {str(e)}")
        return False

def test_voices(api_url, api_key):
    """Test the voices list endpoint."""
    print("\n=== Testing Voices List Endpoint ===")
    
    headers = {"Authorization": f"token {api_key}"}
    
    try:
        response = requests.get(f"{api_url}/voices", headers=headers)
        
        if response.status_code == 200:
            print("Voices list successful!")
            voices = response.json().get("voices", [])
            print(f"Found {len(voices)} voices:")
            
            for voice in voices:
                print(f"  - {voice['name']}: {voice['description']}")
            
            return True
        else:
            print(f"Voices list failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing voices list: {str(e)}")
        return False

def test_rest_api(api_url, api_key, text, voice):
    """Test the REST API endpoint."""
    print("\n=== Testing REST API ===")
    
    headers = {"Authorization": f"token {api_key}"}
    data = {
        "text": text,
        "voice": voice,
        "encoding": "wav",
        "sample_rate": 24000,
        "language": "ja"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/speak")
        print(f"Text: '{text}'")
        print(f"Voice: {voice}")
        
        response = requests.post(f"{api_url}/v1/speak", headers=headers, json=data)
        
        if response.status_code == 200:
            print("REST API request successful!")
            
            # Save the audio response
            output_file = f"rest_api_test_{voice}.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            print(f"Audio saved to {output_file}")
            
            # Try to play the audio if pydub is available
            try:
                audio = AudioSegment.from_file(BytesIO(response.content), format="wav")
                print("Playing audio...")
                play(audio)
            except Exception as e:
                print(f"Note: Could not play audio: {str(e)}")
            
            return True
        else:
            print(f"REST API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing REST API: {str(e)}")
        return False

async def test_websocket_api(api_url, api_key, text, voice):
    """Test the WebSocket API endpoint."""
    print("\n=== Testing WebSocket API ===")
    
    # Convert HTTP URL to WebSocket URL
    if api_url.startswith("http://"):
        ws_url = f"ws://{api_url[7:]}/v1/speak"
    elif api_url.startswith("https://"):
        ws_url = f"wss://{api_url[8:]}/v1/speak"
    else:
        ws_url = f"ws://{api_url}/v1/speak"
    
    # Prepare parameters for request
    data = {
        "text": text,
        "voice": voice,
        "encoding": "wav",
        "sample_rate": 24000,
        "language": "ja"
    }

    # Add query parameters
    ws_url += f"?encoding=wav&sample_rate=24000"
    
    headers = {"Authorization": f"token {api_key}"}
    
    try:
        print(f"Connecting to WebSocket at {ws_url}")
        
        async with websockets.connect(ws_url, extra_headers=headers) as websocket:
            print("WebSocket connection established!")
            
            # Create a message
            message = {
                "type": "Speak",
                "text": text,
                "voice": voice,
                "language": "ja"
            }
            
            print(f"Sending message: {json.dumps(message)}")
            await websocket.send(json.dumps(message))
            
            # Receive and save audio data
            output_file = f"websocket_test_{voice}.wav"
            audio_data = b""
            
            print("Waiting for audio data...")
            try:
                # Set a timeout for receiving data
                audio_data = await asyncio.wait_for(websocket.recv(), timeout=30)
                
                # If we received binary data, save it
                if isinstance(audio_data, bytes):
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    
                    print(f"Audio saved to {output_file}")
                    
                    # Try to play the audio if pydub is available
                    try:
                        audio = AudioSegment.from_file(BytesIO(audio_data), format="wav")
                        print("Playing audio...")
                        play(audio)
                    except Exception as e:
                        print(f"Note: Could not play audio: {str(e)}")
                    
                    return True
                else:
                    print(f"Received non-binary data: {audio_data}")
                    return False
            except asyncio.TimeoutError:
                print("Timeout waiting for audio data")
                return False
    except Exception as e:
        print(f"Error testing WebSocket API: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the Deepgram Mock API")
    parser.add_argument("-u", "--url", type=str, default=DEFAULT_API_URL,
                      help=f"API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("-k", "--api-key", type=str, default=DEFAULT_API_KEY,
                      help="API key for authentication")
    parser.add_argument("-t", "--text", type=str, default="Hello, this is a test of the Deepgram API with GPT-SoVITS.",
                      help="Text to synthesize")
    parser.add_argument("-v", "--voice", type=str, default="nova",
                      help="Voice to use for synthesis")
    
    args = parser.parse_args()
    
    # Run the health check test
    if not test_health(args.url, args.api_key):
        print("Health check failed, exiting.")
        sys.exit(1)
    
    # Run the voices list test
    test_voices(args.url, args.api_key)
    
    # Run the REST API test
    test_rest_api(args.url, args.api_key, args.text, args.voice)
    
    # Run the WebSocket API test
    asyncio.run(test_websocket_api(args.url, args.api_key, args.text, args.voice))
    
    print("\n=== Test Summary ===")
    print("All tests completed. Check the outputs for results.")

if __name__ == "__main__":
    main() 