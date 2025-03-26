import requests
import argparse
import json
import time
import os
import wave
import numpy as np
from io import BytesIO
import soundfile as sf

def test_tts_get(base_url, text, ref_audio_path, text_lang="ja", prompt_lang="ja", prompt_text="", save_path="test_output.wav"):
    """Test the TTS GET endpoint"""
    url = f"{base_url}/tts"
    
    params = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False
    }
    
    print(f"Making GET request to {url} with params: {params}")
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            print(f"GET request successful. Status code: {response.status_code}")
            # Save the audio data
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Audio saved to {save_path}")
            return True
        else:
            print(f"GET request failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error during GET request: {str(e)}")
        return False

def test_tts_post(base_url, text, ref_audio_path, text_lang="ja", prompt_lang="ja", prompt_text="", save_path="test_output_post.wav"):
    """Test the TTS POST endpoint"""
    url = f"{base_url}/tts"
    
    data = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "text_split_method": "cut5",
        "batch_size": 1,
    }
    
    print(f"Making POST request to {url} with data: {json.dumps(data, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print(f"POST request successful. Status code: {response.status_code}")
            # Save the audio data
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Audio saved to {save_path}")
            return True
        else:
            print(f"POST request failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error during POST request: {str(e)}")
        return False

def test_streaming_tts(base_url, text, ref_audio_path, text_lang="ja", prompt_lang="ja", prompt_text="", save_path="test_streaming.wav"):
    """Test the TTS streaming mode"""
    url = f"{base_url}/tts"
    
    params = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_lang": prompt_lang,
        "prompt_text": prompt_text,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": True
    }
    
    print(f"Making streaming request to {url}")
    
    try:
        with requests.get(url, params=params, stream=True) as response:
            if response.status_code == 200:
                print(f"Streaming request successful. Status code: {response.status_code}")
                
                # Save the streaming audio
                audio_data = BytesIO()
                for chunk in response.iter_content(chunk_size=1024):
                    audio_data.write(chunk)
                
                with open(save_path, "wb") as f:
                    f.write(audio_data.getvalue())
                print(f"Streaming audio saved to {save_path}")
                return True
            else:
                print(f"Streaming request failed. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
    except Exception as e:
        print(f"Error during streaming request: {str(e)}")
        return False

def test_model_switching(base_url):
    """Test the model switching endpoints"""
    # Test GPT weights switching
    gpt_url = f"{base_url}/set_gpt_weights"
    gpt_params = {
        "weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    }
    
    try:
        print("Testing GPT weights switching...")
        response = requests.get(gpt_url, params=gpt_params)
        if response.status_code == 200:
            print("GPT weights switching successful!")
        else:
            print(f"GPT weights switching failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error during GPT weights switching: {str(e)}")
    
    # Test SoVits weights switching
    sovits_url = f"{base_url}/set_sovits_weights"
    sovits_params = {
        "weights_path": "GPT_SoVITS/pretrained_models/s2Gv3.pth"
    }
    
    try:
        print("Testing SoVits weights switching...")
        response = requests.get(sovits_url, params=sovits_params)
        if response.status_code == 200:
            print("SoVits weights switching successful!")
        else:
            print(f"SoVits weights switching failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error during SoVits weights switching: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test the GPT-SoVITS API")
    parser.add_argument("-a", "--address", type=str, default="127.0.0.1", help="API server address")
    parser.add_argument("-p", "--port", type=int, default=9880, help="API server port")
    parser.add_argument("-r", "--ref_audio", type=str, required=True, help="Reference audio path")
    parser.add_argument("-o", "--output_dir", type=str, default="test_outputs", help="Output directory for test audio files")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.address}:{args.port}"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define test text
    test_text = "よろしくお願いします！"
    
    print(f"Testing GPT-SoVITS API at {base_url}")
    print(f"Using reference audio: {args.ref_audio}")
    
    # Run the tests
    success_count = 0
    total_tests = 3
    
    # Test 1: GET request
    get_output_path = os.path.join(args.output_dir, "get_test.wav")
    if test_tts_get(base_url, test_text, args.ref_audio, save_path=get_output_path):
        success_count += 1
    
    # Test 2: POST request
    post_output_path = os.path.join(args.output_dir, "post_test.wav")
    if test_tts_post(base_url, test_text, args.ref_audio, save_path=post_output_path):
        success_count += 1
    
    # Test 3: Streaming request
    streaming_output_path = os.path.join(args.output_dir, "streaming_test.wav")
    if test_streaming_tts(base_url, test_text, args.ref_audio, save_path=streaming_output_path):
        success_count += 1
    
    # Test model switching (not counted in success total as it might not be needed)
    test_model_switching(base_url)
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {success_count}")
    print(f"Failed tests: {total_tests - success_count}")
    
    if success_count == total_tests:
        print("\nAll tests passed! The API is working correctly.")
    else:
        print(f"\n{total_tests - success_count} tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 