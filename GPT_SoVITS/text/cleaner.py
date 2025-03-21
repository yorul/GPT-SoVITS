from text import cleaned_text_to_sequence
import os
# if os.environ.get("version","v1")=="v1":
#     from text import chinese
#     from text.symbols import symbols
# else:
#     from text import chinese2 as chinese
#     from text.symbols2 import symbols

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text(text, language, version=None):
    if version is None:version=os.environ.get('version', 'v2')
    print(f"\n--- Cleaner Debug ---")
    print(f"Input text: '{text}'")
    print(f"Language: {language}")
    print(f"Version: {version}")
    
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean","yue":"cantonese"}

    if(language not in language_module_map):
        language="en"
        text=" "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)
    
    language_module = __import__("text."+language_module_map[language],fromlist=[language_module_map[language]])
    if hasattr(language_module,"text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text=text
    
    print(f"Normalized text: '{norm_text}'")
    
    if language == "zh" or language=="yue":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "ja":  # Specific handling for Japanese
        phones, word2ph = language_module.g2p(norm_text)
        
        # Check for and fix length mismatches
        text_no_spaces = norm_text.replace(" ", "")
        if len(text_no_spaces) != len(word2ph):
            print(f"WARNING: Length mismatch between text ({len(text_no_spaces)}) and word2ph ({len(word2ph)})")
            if len(word2ph) < len(text_no_spaces):
                # Extend word2ph
                word2ph = word2ph + [0] * (len(text_no_spaces) - len(word2ph))
            else:
                # Truncate word2ph
                word2ph = word2ph[:len(text_no_spaces)]
            print(f"Fixed word2ph length to {len(word2ph)}")
        
        # Check for and fix phone count mismatches
        if sum(word2ph) != len(phones):
            print(f"WARNING: Phone count mismatch between word2ph sum ({sum(word2ph)}) and phones ({len(phones)})")
            if sum(word2ph) < len(phones):
                # Add remaining phones to last character
                if len(word2ph) > 0:
                    word2ph[-1] += len(phones) - sum(word2ph)
            else:
                # Remove excess phones from last character
                excess = sum(word2ph) - len(phones)
                if word2ph[-1] > excess:
                    word2ph[-1] -= excess
                else:
                    # We need to reduce across multiple characters
                    for i in range(len(word2ph)-1, -1, -1):
                        if word2ph[i] > 0:
                            if word2ph[i] >= excess:
                                word2ph[i] -= excess
                                break
                            else:
                                excess -= word2ph[i]
                                word2ph[i] = 0
            print(f"Fixed word2ph sum to {sum(word2ph)}")
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    
    print(f"After g2p - Phones: {phones}")
    print(f"After g2p - word2ph: {word2ph}")
    
    phones = ['UNK' if ph not in symbols else ph for ph in phones]
    
    print(f"Final phones after UNK check: {phones}")
    print(f"Final word2ph: {word2ph}")
    
    # Final verification
    if language == "ja" and word2ph is not None:
        text_no_spaces = norm_text.replace(" ", "")
        if len(text_no_spaces) != len(word2ph):
            print(f"ERROR: Length mismatch between text ({len(text_no_spaces)}) and word2ph ({len(word2ph)})")
        if sum(word2ph) != len(phones):
            print(f"ERROR: Phone count mismatch between word2ph sum ({sum(word2ph)}) and phones ({len(phones)})")
    
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol, version=None):
    if version is None:version=os.environ.get('version', 'v2')
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean","yue":"cantonese"}

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("text."+language_module_map[language],fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language, version=None):
    version = os.environ.get('version',version)
    if version is None:version='v2'
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones, version)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
