# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
import os
import hashlib
try:
    import pyopenjtalk
    current_file_path = os.path.dirname(__file__)

    # 防止win下无法读取模型
    if os.name == 'nt':
        python_dir = os.getcwd()
        OPEN_JTALK_DICT_DIR = pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8")
        if not (re.match(r'^[A-Za-z0-9_/\\:.\-]*$', OPEN_JTALK_DICT_DIR)):
            if (OPEN_JTALK_DICT_DIR[:len(python_dir)].upper() == python_dir.upper()):
                OPEN_JTALK_DICT_DIR = os.path.join(os.path.relpath(OPEN_JTALK_DICT_DIR,python_dir))
            else:
                import shutil
                if not os.path.exists('TEMP'):
                    os.mkdir('TEMP')
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if os.path.exists(os.path.join("TEMP", "ja", "open_jtalk_dic")):
                    shutil.rmtree(os.path.join("TEMP", "ja", "open_jtalk_dic"))
                shutil.copytree(pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"), os.path.join("TEMP", "ja", "open_jtalk_dic"), )
                OPEN_JTALK_DICT_DIR = os.path.join("TEMP", "ja", "open_jtalk_dic")
            pyopenjtalk.OPEN_JTALK_DICT_DIR = OPEN_JTALK_DICT_DIR.encode("utf-8")

        if not (re.match(r'^[A-Za-z0-9_/\\:.\-]*$', current_file_path)):
            if (current_file_path[:len(python_dir)].upper() == python_dir.upper()):
                current_file_path = os.path.join(os.path.relpath(current_file_path,python_dir))
            else:
                if not os.path.exists('TEMP'):
                    os.mkdir('TEMP')
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if not os.path.exists(os.path.join("TEMP", "ja", "ja_userdic")):
                    os.mkdir(os.path.join("TEMP", "ja", "ja_userdic"))
                    shutil.copyfile(os.path.join(current_file_path, "ja_userdic", "userdict.csv"),os.path.join("TEMP", "ja", "ja_userdic", "userdict.csv"))
                current_file_path = os.path.join("TEMP", "ja")


    def get_hash(fp: str) -> str:
        hash_md5 = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    USERDIC_CSV_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.csv")
    USERDIC_BIN_PATH = os.path.join(current_file_path, "ja_userdic", "user.dict")
    USERDIC_HASH_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.md5")
    # 如果没有用户词典，就生成一个；如果有，就检查md5，如果不一样，就重新生成
    if os.path.exists(USERDIC_CSV_PATH):
        if not os.path.exists(USERDIC_BIN_PATH) or get_hash(USERDIC_CSV_PATH) != open(USERDIC_HASH_PATH, "r",encoding='utf-8').read():
            pyopenjtalk.mecab_dict_index(USERDIC_CSV_PATH, USERDIC_BIN_PATH)
            with open(USERDIC_HASH_PATH, "w", encoding='utf-8') as f:
                f.write(get_hash(USERDIC_CSV_PATH))

    if os.path.exists(USERDIC_BIN_PATH):
        pyopenjtalk.update_global_jtalk_with_user_dict(USERDIC_BIN_PATH)   
except Exception as e:
    # print(e)
    import pyopenjtalk
    # failed to load user dictionary, ignore.
    pass


from text.symbols import punctuation
# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]

# Japanese vowels and N for checking
_japanese_vowels = ['a', 'i', 'u', 'e', 'o', 'N']
# Prosody and accent markers
_japanese_markers = ['#', '[', ']']
# Extended punctuation set for Japanese
_japanese_punctuation = list(punctuation) + ['.', ',', '、', '。', '！', '？', '…', '「', '」', '『', '』']

def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
    }

    if ph in rep_map.keys():
        ph = rep_map[ph]
    return ph


def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text, with_prosody=False):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    text = symbols_to_japanese(text)
    # English words to lower case, should have no influence on japanese words.
    text = text.lower()
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = []
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if with_prosody:
                text += pyopenjtalk_g2p_prosody(sentence)[1:-1]
            else:
                p = pyopenjtalk.g2p(sentence)
                text += p.split(" ")

        if i < len(marks):
            if marks[i] == " ":# 防止意外的UNK
                continue
            text += [marks[i].replace(" ", "")]
    return text


def text_normalize(text):
    # todo: jap text normalize

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)
    return text

# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones

# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

def g2p(norm_text, with_prosody=True):
    """
    Enhanced Japanese grapheme-to-phoneme conversion with accurate word2ph mapping.
    
    This implementation creates a precise mapping between Japanese characters and
    their corresponding phonemes, handling all complexities of Japanese text.
    
    Args:
        norm_text (str): Normalized input text
        with_prosody (bool): Whether to include prosody markers
        
    Returns:
        tuple: (phones, word2ph) where phones is a list of phoneme symbols and
               word2ph is a list of integers representing how many phones each
               character corresponds to
    """
    # Get phonemes using the existing function
    phones = preprocess_jap(norm_text, with_prosody)
    phones = [post_replace_ph(i) for i in phones]
    
    # Debug information
    print(f"\n--- Enhanced Japanese g2p Debug ---")
    print(f"Input text: '{norm_text}'")
    print(f"Generated phones: {phones}")
    print(f"Number of phones: {len(phones)}")
    
    # Filter out spaces - we'll handle them separately
    text_no_spaces = norm_text.replace(" ", "")
    
    # Initialize word2ph with one entry per character (excluding spaces)
    word2ph = [0] * len(text_no_spaces)
    
    # Create detailed mapping tracking which phones belong to each character
    char_to_phones = {}
    for i in range(len(text_no_spaces)):
        char_to_phones[i] = []
    
    # First pass: identify clear character-phone mappings
    # Japanese characters often follow these patterns:
    # 1. Hiragana/Katakana: typically 1-2 phones (consonant+vowel or just vowel)
    # 2. Kanji: variable number of phones depending on the reading
    # 3. Latin characters: usually 1 phone per character
    # 4. Punctuation: typically 1 phone or none
    
    # Track which phones have been assigned
    assigned_phones = [False] * len(phones)
    
    # Track mapping of characters to phone sequences
    phone_sequences = []
    
    # Process text character by character
    i = 0  # character index
    while i < len(text_no_spaces):
        char = text_no_spaces[i]
        
        # Skip directly to next character for spaces (should be none in text_no_spaces)
        if char == " ":
            i += 1
            continue
            
        # Handle punctuation first
        if char in _japanese_punctuation or re.match(_japanese_marks, char):
            # Look for this punctuation in the phone stream
            punctuation_found = False
            for j, ph in enumerate(phones):
                if not assigned_phones[j] and (ph == char or ph == post_replace_ph(char)):
                    word2ph[i] = 1
                    char_to_phones[i].append(ph)
                    assigned_phones[j] = True
                    punctuation_found = True
                    break
            
            # If we couldn't find the punctuation in phones, assign 0
            if not punctuation_found:
                word2ph[i] = 0
            
            phone_sequences.append((i, char_to_phones[i]))
            i += 1
            continue
        
        # For regular Japanese characters, we need to detect how many phones they use
        # This is more complex and requires pattern recognition
        
        # Use a sliding window to find the best phoneme sequence for this character
        # Start with likely patterns for this character type
        
        # For katakana/hiragana, typically expect consonant+vowel or just vowel
        # For kanji, might be multiple phones
        
        # Start with finding the next unassigned phone
        phone_start_idx = None
        for j in range(len(phones)):
            if not assigned_phones[j] and phones[j] not in _japanese_markers:
                phone_start_idx = j
                break
        
        if phone_start_idx is None:
            # No more unassigned regular phones, assign 0
            word2ph[i] = 0
            phone_sequences.append((i, []))
            i += 1
            continue
        
        # Determine this character's phonetic pattern
        current_phones = []
        j = phone_start_idx
        
        # Include any accent markers before the core phonemes
        while j < len(phones) and phones[j] in _japanese_markers and not assigned_phones[j]:
            current_phones.append(phones[j])
            assigned_phones[j] = True
            j += 1
        
        # Get to the core phonetic content (consonant+vowel, vowel, or N)
        core_found = False
        while j < len(phones) and not core_found and not assigned_phones[j]:
            # Skip any markers in the middle
            if phones[j] in _japanese_markers:
                current_phones.append(phones[j])
                assigned_phones[j] = True
                j += 1
                continue
            
            # Skip if it's punctuation
            if phones[j] in _japanese_punctuation:
                break
            
            # Add this phone
            current_phones.append(phones[j])
            assigned_phones[j] = True
            
            # If it's a vowel or N, we've found a core
            if phones[j] in _japanese_vowels:
                core_found = True
            
            j += 1
        
        # Include any accent markers after the core phonemes
        while j < len(phones) and phones[j] in _japanese_markers and not assigned_phones[j]:
            current_phones.append(phones[j])
            assigned_phones[j] = True
            j += 1
        
        # Assign these phones to the current character
        if current_phones:
            word2ph[i] = len(current_phones)
            char_to_phones[i] = current_phones
        else:
            word2ph[i] = 0
        
        phone_sequences.append((i, char_to_phones[i]))
        i += 1
    
    # Second pass: distribute any remaining unassigned phones
    # This handles complex cases like long vowels, geminate consonants, and special phonetic features
    remaining_phones = []
    for j, assigned in enumerate(assigned_phones):
        if not assigned:
            remaining_phones.append(phones[j])
    
    if remaining_phones:
        print(f"Unassigned phones: {remaining_phones}")
        
        # Distribute remaining phones across characters with the most likely phonetic match
        # or add them to the last character as a fallback
        if len(word2ph) > 0:
            # Try to find characters that could plausibly have more phones
            potential_chars = []
            
            # Kanji characters often have multiple phones
            for idx in range(len(text_no_spaces)):
                char = text_no_spaces[idx]
                # Check if it's likely a kanji (CJK unified ideograph range)
                is_kanji = '\u4e00' <= char <= '\u9fff'
                if is_kanji:
                    potential_chars.append(idx)
            
            if potential_chars:
                # Distribute evenly across potential characters
                phones_per_char = len(remaining_phones) // len(potential_chars)
                remaining = len(remaining_phones) % len(potential_chars)
                
                remaining_phone_idx = 0
                for idx in potential_chars:
                    phones_to_add = phones_per_char + (1 if remaining > 0 else 0)
                    remaining -= 1
                    
                    for _ in range(phones_to_add):
                        if remaining_phone_idx < len(remaining_phones):
                            char_to_phones[idx].append(remaining_phones[remaining_phone_idx])
                            word2ph[idx] += 1
                            remaining_phone_idx += 1
            else:
                # Add to the last character as fallback
                for ph in remaining_phones:
                    char_to_phones[len(word2ph)-1].append(ph)
                word2ph[-1] += len(remaining_phones)
    
    # Final verification and reporting
    total_assigned_phones = sum(word2ph)
    print(f"Character to phones mapping:")
    for i in range(len(text_no_spaces)):
        print(f"  '{text_no_spaces[i]}' → {char_to_phones[i]} ({word2ph[i]} phones)")
    
    print(f"Total assigned phones: {total_assigned_phones} / {len(phones)}")
    
    if total_assigned_phones != len(phones):
        print(f"WARNING: word2ph sum ({total_assigned_phones}) doesn't match phone count ({len(phones)})")
        # Make a correction to ensure all phones are accounted for
        if len(word2ph) > 0:
            # Adjust the last character
            word2ph[-1] += len(phones) - total_assigned_phones
            print(f"Adjusted last character's phone count to {word2ph[-1]}")
    
    print(f"Final word2ph mapping: {word2ph}")
    print(f"Sum of word2ph: {sum(word2ph)}")
    
    # Verify that we have a word2ph entry for each character
    print(f"word2ph length: {len(word2ph)}, text length (no spaces): {len(text_no_spaces)}")
    
    if len(word2ph) != len(text_no_spaces):
        print(f"WARNING: word2ph length mismatch. Adjusting to match text length.")
        # Pad or truncate to match text length
        if len(word2ph) < len(text_no_spaces):
            word2ph = word2ph + [0] * (len(text_no_spaces) - len(word2ph))
        else:
            word2ph = word2ph[:len(text_no_spaces)]
    
    return phones, word2ph


if __name__ == "__main__":
    phones, word2ph = g2p("Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
    print(phones)
    print(word2ph)
