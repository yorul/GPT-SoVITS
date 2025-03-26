import os
import json
import time
import base64
import asyncio
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
from io import BytesIO

import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Request, Depends, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import soundfile as sf
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
GPT_SOVITS_API_URL = os.environ.get("GPT_SOVITS_API_URL", "http://127.0.0.1:9880")
API_KEY = os.environ.get("API_KEY", "default-api-key")
VOICE_MAPPING_FILE = os.environ.get("VOICE_MAPPING_FILE", "voice_mapping.json")
DEFAULT_MODEL = "aura-asteria-ja"
DEFAULT_VOICE = "nova"
DEFAULT_ENCODING = "wav"
DEFAULT_RATE = 24000
DEFAULT_CONTAINER = "none"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load voice and language mappings from JSON file
try:
    with open(VOICE_MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
        VOICE_MAPPING = {k: v["ref_audio_path"] for k, v in mapping_data.get("voices", {}).items()}
        LANGUAGE_MAPPING = mapping_data.get("languages", {})
        
        logger.info(f"Loaded voice mappings for {len(VOICE_MAPPING)} voices from {VOICE_MAPPING_FILE}")
except Exception as e:
    logger.warning(f"Failed to load voice mappings from {VOICE_MAPPING_FILE}: {str(e)}")
    logger.warning("Using default voice mappings")
    
    # Default voice mapping if file not found
    VOICE_MAPPING = {
        "nova": "nova_reference.wav",
        "aura": "aura_reference.wav",
        "stella": "stella_reference.wav",
        "onyx": "onyx_reference.wav",
        "shimmer": "shimmer_reference.wav",
        "default": "default_reference.wav"
    }
    
    # Default language mapping
    LANGUAGE_MAPPING = {
        "ja": "ja",
        "ja-JP": "ja",
        "auto": "auto"
    }

def validate_api_key(authorization: str = Header(None)):
    """Validate the API key."""
    if not authorization:
        raise HTTPException(status_code=401, detail="No API key provided")
    
    # Extract API key from the authorization header (format: "token YOUR_API_KEY")
    if authorization.startswith("token "):
        api_key = authorization.split(" ")[1]
    else:
        api_key = authorization
    
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return api_key

def get_voice_reference(voice: str) -> str:
    """Get the reference audio path for a given Deepgram voice."""
    return VOICE_MAPPING.get(voice, VOICE_MAPPING.get("default", "default_reference.wav"))

def get_language(language: str) -> str:
    """Map Deepgram language to GPT-SoVITS language."""
    if not language:
        return "ja"
    
    # First try exact match
    if language in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[language]
    
    # Then try lowercase match
    language_lower = language.lower()
    if language_lower in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[language_lower]
    
    # Then try to match prefix (e.g., 'ja-JP' matches 'ja')
    for lang_prefix in LANGUAGE_MAPPING:
        if language_lower.startswith(lang_prefix.lower() + "-"):
            return LANGUAGE_MAPPING[lang_prefix]
    
    # If all else fails, return safe default
    return "ja"

async def text_to_audio(
    text: str,
    voice: str,
    encoding: str = DEFAULT_ENCODING,
    sample_rate: int = DEFAULT_RATE,
    language: str = "ja",
    streaming: bool = False,
    prompt_text: str = None
) -> bytes:
    """Convert text to audio using GPT-SoVITS API."""
    # Get the voice reference path
    ref_audio_path = get_voice_reference(voice)
    
    # Always use Japanese for language
    text_lang = "ja"
    
    # If prompt_text is not provided, get it from the mapping file
    if prompt_text is None:
        try:
            if os.path.exists(VOICE_MAPPING_FILE):
                with open(VOICE_MAPPING_FILE, 'r') as f:
                    mapping_data = json.load(f)
                    voices = mapping_data.get("voices", {})
                    if voice in voices and "prompt_text" in voices[voice]:
                        prompt_text = voices[voice]["prompt_text"]
                    elif "default" in voices and "prompt_text" in voices["default"]:
                        prompt_text = voices["default"]["prompt_text"]
        except Exception as e:
            logger.warning(f"Failed to load prompt text: {str(e)}")
    
    # If still None, use an empty string
    if prompt_text is None:
        prompt_text = ""
    
    logger.info(f"Using reference audio: {ref_audio_path}")
    logger.info(f"Using prompt text: {prompt_text}")
    logger.info(f"Using text language: {text_lang}")
    
    # Prepare parameters for GPT-SoVITS API
    # Always request WAV format from GPT-SoVITS for best compatibility
    params = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_lang": text_lang,
        "prompt_text": prompt_text,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",  # Always request WAV format
        "streaming_mode": streaming
    }
    
    try:
        # Log the parameters being sent to GPT-SoVITS
        logger.info(f"Sending to GPT-SoVITS API: {params}")
        
        # Call GPT-SoVITS API
        if streaming:
            # For streaming, we need to establish a request with stream=True
            response = requests.get(f"{GPT_SOVITS_API_URL}/tts", params=params, stream=True)
        else:
            # Regular request for non-streaming
            response = requests.get(f"{GPT_SOVITS_API_URL}/tts", params=params)
        
        if response.status_code != 200:
            logger.error(f"GPT-SoVITS API error: {response.text}")
            raise HTTPException(status_code=500, detail="TTS service error")
        
        # Convert to the requested encoding if needed
        audio_data = response.content
        if encoding.lower() != "wav":
            # We'll implement format conversion in a separate function
            audio_data = convert_audio_format(audio_data, encoding)
        
        return audio_data
    except Exception as e:
        logger.error(f"Error calling GPT-SoVITS API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS service error: {str(e)}")

def convert_audio_format(audio_data: bytes, target_format: str) -> bytes:
    """Convert audio data from WAV to another format."""
    try:
        # Create a BytesIO object from the audio data
        audio_io = BytesIO(audio_data)
        
        # Read the WAV data
        data, samplerate = sf.read(audio_io)
        
        # Create a BytesIO object for the output
        output = BytesIO()
        
        if target_format.lower() == "mp3":
            # Write WAV to the BytesIO object as we don't have direct MP3 support
            # This will be served as MP3 but remains WAV internally
            sf.write(output, data, samplerate, format='wav')
        elif target_format.lower() == "ogg":
            # Write OGG to the BytesIO object
            sf.write(output, data, samplerate, format='ogg')
        else:
            # Default to WAV
            sf.write(output, data, samplerate, format='wav')
        
        # Get the bytes from the BytesIO object
        output.seek(0)
        return output.read()
    except Exception as e:
        logger.error(f"Error converting audio format: {str(e)}")
        # Return original data as fallback
        return audio_data

@app.get("/voices")
async def list_voices(authorization: str = Depends(validate_api_key)):
    """Return a list of available voices."""
    try:
        if os.path.exists(VOICE_MAPPING_FILE):
            with open(VOICE_MAPPING_FILE, 'r') as f:
                mapping_data = json.load(f)
                voices = mapping_data.get("voices", {})
                
                # Format the response to match Deepgram's API
                result = []
                for voice_id, voice_data in voices.items():
                    if voice_id == "default":
                        continue
                        
                    result.append({
                        "id": voice_id,
                        "name": voice_id.capitalize(),
                        "description": voice_data.get("description", ""),
                        "language": "ja"  # Always use Japanese
                    })
                
                return JSONResponse(content={"voices": result})
        else:
            # Return default voices if file not found
            default_voices = []
            for voice_id in VOICE_MAPPING:
                if voice_id == "default":
                    continue
                    
                default_voices.append({
                    "id": voice_id,
                    "name": voice_id.capitalize(),
                    "description": f"{voice_id.capitalize()} voice",
                    "language": "ja"  # Always use Japanese
                })
                
            return JSONResponse(content={"voices": default_voices})
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/v1/speak")
async def speak(
    request: Request,
    authorization: str = Depends(validate_api_key)
):
    """REST API endpoint that mimics Deepgram's /v1/speak endpoint."""
    try:
        # Parse request body
        data = await request.json()
        
        # Extract parameters from request
        text = data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        voice = data.get("voice", DEFAULT_VOICE)
        encoding = data.get("encoding", DEFAULT_ENCODING)
        sample_rate = int(data.get("sample_rate", DEFAULT_RATE))
        model = data.get("model", DEFAULT_MODEL)
        language = "ja"  # Always use Japanese
        prompt_text = data.get("prompt_text", None)  # Allow prompt_text to be specified
        
        # Convert text to audio
        audio_data = await text_to_audio(
            text=text,
            voice=voice,
            encoding=encoding,
            sample_rate=sample_rate,
            language=language,
            streaming=False,
            prompt_text=prompt_text
        )
        
        # Return audio data
        return StreamingResponse(
            content=BytesIO(audio_data),
            media_type=f"audio/{encoding}" if encoding.lower() != "pcm" else "application/octet-stream",
            headers={
                "X-RequestID": str(time.time()),
                "Content-Length": str(len(audio_data))
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in speak endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.websocket("/v1/speak")
async def speak_ws(
    websocket: WebSocket,
    encoding: str = Query(DEFAULT_ENCODING),
    model: str = Query(DEFAULT_MODEL),
    sample_rate: int = Query(DEFAULT_RATE),
    container: str = Query(DEFAULT_CONTAINER),
):
    """WebSocket endpoint that mimics Deepgram's WebSocket interface."""
    try:
        await websocket.accept()
        
        # WebSocket messages will be processed asynchronously
        active = True
        
        # Validate API key from WebSocket headers
        headers = dict(websocket.headers)
        authorization = headers.get("authorization", None)
        if not authorization:
            await websocket.close(1008, "No API key provided")
            return
        
        try:
            validate_api_key(authorization)
        except HTTPException:
            await websocket.close(1008, "Invalid API key")
            return
        
        async def receive_messages():
            nonlocal active
            while active:
                try:
                    message = await websocket.receive_text()
                    message_data = json.loads(message)
                    
                    # Process different types of messages
                    if message_data.get("type") == "Speak":
                        text = message_data.get("text", "")
                        if not text:
                            continue
                        
                        voice = message_data.get("voice", DEFAULT_VOICE)
                        language = "ja"  # Always use Japanese
                        prompt_text = message_data.get("prompt_text", None)  # Allow prompt_text to be specified
                        
                        # Convert text to audio (streaming mode)
                        audio_data = await text_to_audio(
                            text=text,
                            voice=voice,
                            encoding=encoding,
                            sample_rate=sample_rate,
                            language=language,
                            streaming=True,
                            prompt_text=prompt_text
                        )
                        
                        # Send audio data as binary
                        if encoding.lower() != "pcm":
                            await websocket.send_bytes(audio_data)
                        else:
                            # For PCM, we need to encode as base64 strings
                            await websocket.send_text(base64.b64encode(audio_data).decode('utf-8'))
                    
                    elif message_data.get("type") == "CloseStream":
                        active = False
                        await websocket.close()
                        break
                    
                    elif message_data.get("type") == "ControlMessage":
                        # Handle control messages (like pausing/resuming)
                        control_type = message_data.get("control", {}).get("type", "")
                        if control_type == "pause":
                            # Implement pause logic if needed
                            pass
                        elif control_type == "resume":
                            # Implement resume logic if needed
                            pass
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {str(e)}")
                    await websocket.send_json({
                        "type": "Error",
                        "error": str(e)
                    })
        
        # Start processing messages
        await receive_messages()
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(1011, f"Server error: {str(e)}")
        except:
            pass

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if GPT-SoVITS API is accessible with a minimal valid request
        # Use the default voice and minimal required parameters
        minimal_params = {
            "text": "テスト",
            "text_lang": "ja",
            "ref_audio_path": get_voice_reference(DEFAULT_VOICE),
            "prompt_lang": "ja",
            "prompt_text": "",
            "media_type": "wav"
        }
        
        response = requests.get(f"{GPT_SOVITS_API_URL}/tts", params=minimal_params, timeout=2)
        gpt_sovits_status = "ok" if response.status_code == 200 else "error"
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        gpt_sovits_status = "error"
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "gpt_sovits_api": gpt_sovits_status
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deepgram Mock API for GPT-SoVITS (Japanese Only)")
    parser.add_argument("-a", "--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("-u", "--gpt-sovits-url", type=str, default=GPT_SOVITS_API_URL, 
                       help=f"GPT-SoVITS API URL (default: {GPT_SOVITS_API_URL})")
    parser.add_argument("-k", "--api-key", type=str, default=API_KEY,
                       help="API key for authentication")
    parser.add_argument("-m", "--mapping-file", type=str, default=VOICE_MAPPING_FILE,
                       help="Path to voice mapping JSON file")
    
    args = parser.parse_args()
    
    # Update global variables
    GPT_SOVITS_API_URL = args.gpt_sovits_url
    API_KEY = args.api_key
    VOICE_MAPPING_FILE = args.mapping_file
    
    # Print startup information
    logger.info(f"Starting Deepgram Mock API on {args.host}:{args.port}")
    logger.info(f"Using GPT-SoVITS API at {GPT_SOVITS_API_URL}")
    logger.info(f"Voice mapping file: {VOICE_MAPPING_FILE}")
    
    # Start the FastAPI server
    uvicorn.run(app, host=args.host, port=args.port) 