# -*- coding: utf-8 -*-

import os

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
bert_pretrained_dir = os.environ.get("bert_pretrained_dir")
import torch
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get('version', None)
import sys, numpy as np, traceback, pdb
import os.path
from glob import glob
from tqdm import tqdm
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from tools.my_utils import clean_path

# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]#i_gpu
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name
# bert_pretrained_dir="/data/docker/liujing04/bert-vits2/Bert-VITS2-master20231106/bert/chinese-roberta-wwm-ext-large"

from time import time as ttime
import shutil
import MeCab


def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))


txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
if os.path.exists(txt_path) == False:
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    if os.path.exists(bert_pretrained_dir):...
    else:raise FileNotFoundError(bert_pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):
        # Add logging for debugging
        print("\n--- BERT Feature Debug ---")
        print(f"Input text: '{text}'")
        
        # Handle case when word2ph is None
        if word2ph is None:
            print("WARNING: word2ph is None, creating a default mapping")
            # Create a default mapping with 1 phone per character
            word2ph = [1] * len(text.replace(" ", ""))
        
        with torch.no_grad():
            # Log BERT tokenization
            inputs = tokenizer(text, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            print(f"BERT tokenization: {tokens}")
            print(f"Token IDs: {inputs['input_ids'][0]}")
            
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            
            print(f"BERT features shape: {res.shape}")
            
            # Log word2ph mapping
            print(f"word2ph: {word2ph}")
            text_no_spaces = text.replace(" ", "")
            print(f"Length of text (no spaces): {len(text_no_spaces)}")
            print(f"Length of word2ph: {len(word2ph)}")

        # Create phone-level features by repeating BERT embeddings according to word2ph
        phone_level_feature = []
        
        # Get the actual length of the res tensor to avoid index out of bounds
        res_len = res.shape[0]
        print(f"res_len: {res_len}")
        
        # Handle the case where BERT tokenization is shorter than character count
        if res_len < len(word2ph):
            print(f"Text is longer than BERT tokens: {len(word2ph) - res_len} extra characters")
            
            # For Japanese, we need a better strategy than simply using the last token
            # We'll try to align the BERT tokens with the text characters
            
            # Get text without spaces for character indexing
            text_no_spaces = text.replace(" ", "")
            
            # Naive mapping of BERT tokens to original text indices
            # Note: This is approximate as BERT tokenization can be complex
            token_start_indices = []
            
            # Find likely start positions of each token in the original text
            running_idx = 0
            for t_idx, token in enumerate(tokens[1:-1]):  # Skip [CLS] and [SEP]
                # Skip special token markers like ##
                clean_token = token.replace("##", "")
                
                # Find this token in the text, starting from our current position
                found = False
                for i in range(running_idx, len(text_no_spaces)):
                    # Check if this position starts with the token
                    if i + len(clean_token) <= len(text_no_spaces):
                        if text_no_spaces[i:i+len(clean_token)] == clean_token:
                            token_start_indices.append(i)
                            running_idx = i + len(clean_token)
                            found = True
                            break
                
                # If we couldn't find it, just use the current running_idx
                if not found:
                    token_start_indices.append(running_idx)
                
            # Now map each character to its corresponding token
            char_to_token = []
            for char_idx in range(len(text_no_spaces)):
                # Find which token this character belongs to
                token_idx = None
                for t_idx, start_idx in enumerate(token_start_indices):
                    if t_idx + 1 < len(token_start_indices):
                        if start_idx <= char_idx < token_start_indices[t_idx + 1]:
                            token_idx = t_idx
                            break
                    else:
                        if start_idx <= char_idx:
                            token_idx = t_idx
                            break
                
                # If we couldn't find a token, use the last token
                if token_idx is None:
                    token_idx = len(token_start_indices) - 1
                
                char_to_token.append(token_idx)
            
            # Check if we have a reasonable mapping
            if len(char_to_token) < len(word2ph):
                # Fill in any missing mappings with the last token
                char_to_token.extend([len(token_start_indices) - 1] * (len(word2ph) - len(char_to_token)))
            
            # Now use this mapping to create phone-level features
            for char_idx, token_idx in enumerate(char_to_token):
                if char_idx < len(word2ph):
                    if token_idx < res_len:
                        # Use the correct token for this character
                        repeat_feature = res[token_idx].repeat(word2ph[char_idx], 1)
                    else:
                        # Fallback to last token if our mapping is off
                        repeat_feature = res[-1].repeat(word2ph[char_idx], 1)
                    
                    phone_level_feature.append(repeat_feature)
        else:
            # If BERT tokenization is longer or equal to character count, use a simpler approach
            for i in range(min(len(word2ph), res_len)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)
        
        # Concatenate all phone-level features
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        
        print(f"Final feature shape: {phone_level_feature.T.shape}")
        print(f"bert_feature.shape[-1]: {phone_level_feature.T.shape[-1]}, len(phones): Unknown (check assertion later)")
        
        return phone_level_feature.T

    def process(data, res):
        for name, text, lan in data:
            try:
                name=clean_path(name)
                name = os.path.basename(name)
                print(name)
                
                # Add logging for debugging
                print(f"\n--- Processing file: {name} ---")
                print(f"Original text: '{text}'")
                print(f"Language: {lan}")
                
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan, version
                )
                
                # Add logging for cleaned text results
                print(f"Normalized text: '{norm_text}'")
                print(f"Phones: {phones}")
                print(f"Phones length: {len(phones)}")
                print(f"word2ph: {word2ph}")
                
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if os.path.exists(path_bert) == False and lan in ["zh", "ja"]:
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    
                    # Add more detailed error message
                    print(f"bert_feature.shape[-1]: {bert_feature.shape[-1]}, len(phones): {len(phones)}")
                    if bert_feature.shape[-1] != len(phones):
                        print("WARNING: Mismatch between bert_feature shape and phones length!")
                        print(f"Feature dimensions: {bert_feature.shape}")
                        print(f"Number of phones: {len(phones)}")
                        # Calculate and print the difference
                        diff = abs(bert_feature.shape[-1] - len(phones))
                        print(f"Difference: {diff} {'more' if bert_feature.shape[-1] > len(phones) else 'fewer'} features than phones")
                    
                    assert bert_feature.shape[-1] == len(phones)
                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)
                phones = " ".join(phones)
                # res.append([name,phones])
                res.append([name, phones, word2ph, norm_text])
            except Exception as e:
                print(f"Error processing {name}: {e}")
                traceback.print_exc()

    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
    }
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            # todo.append([name,text,"zh"])
            if language in language_v1_to_language_v2.keys():
                todo.append(
                    [wav_name, text, language_v1_to_language_v2.get(language, language)]
                )
            else:
                print(f"\033[33m[Waring] The {language = } of {wav_name} is not supported for training.\033[0m")
        except:
            print(line, traceback.format_exc())

    process(todo, res)
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")

def get_accurate_japanese_pronunciation(text):
    # Step 1: Use MeCab to get accurate readings
    mecab = MeCab.Tagger("-Oyomi")
    hiragana_reading = mecab.parse(text).strip()
    
    # Step 2: Convert hiragana to phonemes using a well-defined mapping
    phonemes = []
    word2ph = []
    
    # Process the hiragana reading character by character
    for char in hiragana_reading:
        char_phonemes = hiragana_to_phoneme_map.get(char, ['UNK'])
        phonemes.extend(char_phonemes)
        word2ph.append(len(char_phonemes))
    
    return phonemes, word2ph
