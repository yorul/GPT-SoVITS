import os
import requests
from tqdm import tqdm
import zipfile
from pathlib import Path

MODELS = {
    # GPT-SoVITS v2 models
    "gsv-v2final-pretrained/s2G2333k.pth": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth",
    "gsv-v2final-pretrained/s2D2333k.pth": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2D2333k.pth",
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    
    # GPT-SoVITS v3 models
    "s1v3.ckpt": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt",
    "s2Gv3.pth": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2Gv3.pth",
    "bigvgan_generator.pt": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/models--nvidia--bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt",
    "config.json": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/models--nvidia--bigvgan_v2_24khz_100band_256x/config.json",

    # Chinese BERT model
    "chinese-roberta-wwm-ext-large/config.json": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/tokenizer.json",
    
    # Chinese HuBERT model
    "chinese-hubert-base/config.json": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/config.json",
    "chinese-hubert-base/preprocessor_config.json": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/preprocessor_config.json",
    "chinese-hubert-base/pytorch_model.bin": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/pytorch_model.bin",

    # Japanese HuBERT model
    "japanese-hubert-base/config.json": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/config.json",
    "japanese-hubert-base/preprocessor_config.json": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/preprocessor_config.json",
    "japanese-hubert-base/pytorch_model.bin": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/pytorch_model.bin",
    "japanese-hubert-base/tokenizer.json": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/tokenizer.json",

    # Japanese BERT model
    "japanese-roberta-wwm-ext-large/config.json": "https://huggingface.co/tohoku-nlp/bert-base-japanese-v3/resolve/main/config.json",
    "japanese-roberta-wwm-ext-large/pytorch_model.bin": "https://huggingface.co/tohoku-nlp/bert-base-japanese-v3/resolve/main/pytorch_model.bin",
    "japanese-roberta-wwm-ext-large/tokenizer_config.json": "https://huggingface.co/tohoku-nlp/bert-base-japanese-v3/resolve/main/tokenizer_config.json",
    "japanese-roberta-wwm-ext-large/vocab.txt": "https://huggingface.co/tohoku-nlp/bert-base-japanese-v3/resolve/main/vocab.txt"
}

G2PW_URL = "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip"

def download_file(url, dest_path, desc=None):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    desc = desc or os.path.basename(url)
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def setup_pretrained_models():
    # Create necessary directories
    pretrained_dir = Path("GPT_SoVITS/pretrained_models")
    text_dir = Path("GPT_SoVITS/text")
    
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    # Download pretrained models
    print("Downloading pretrained models...")
    for filepath, url in MODELS.items():
        dest_path = pretrained_dir / filepath
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not dest_path.exists():
            print(f"\nDownloading {filepath}...")
            download_file(url, dest_path)
        else:
            print(f"\n{filepath} already exists, skipping...")
            
    # Download and extract G2PWModel
    g2pw_zip = text_dir / "G2PWModel_1.1.zip"
    g2pw_dir = text_dir / "G2PWModel"
    
    if not g2pw_dir.exists():
        print("\nDownloading G2PWModel...")
        download_file(G2PW_URL, g2pw_zip)
        
        print("\nExtracting G2PWModel...")
        with zipfile.ZipFile(g2pw_zip, 'r') as zip_ref:
            zip_ref.extractall(text_dir)
        
        # Rename the extracted directory
        extracted_dir = text_dir / "G2PWModel_1.1"
        if extracted_dir.exists():
            extracted_dir.rename(g2pw_dir)
            
        # Clean up zip file
        g2pw_zip.unlink()
    else:
        print("\nG2PWModel already exists, skipping...")
    
    print("\nAll pretrained models have been downloaded and set up!")

if __name__ == "__main__":
    setup_pretrained_models() 