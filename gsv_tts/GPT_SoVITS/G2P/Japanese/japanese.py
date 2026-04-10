# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
import os
import sys
import pyopenjtalk
from contextlib import contextmanager


@contextmanager
def suppress_c_stderr():
    # 屏蔽C++底层的警告
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(sys.stderr.fileno())
    try:
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(devnull)
        os.close(old_stderr)


class JapaneseG2P:
    def __init__(self):
        # Regular expression matching Japanese without punctuation marks:
        self._japanese_characters = re.compile(
            r"[A-Za-z\d\u3005\u3040-\u30fa\u30fc-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
        )

        # Regular expression matching non-Japanese characters or punctuation marks:
        self._japanese_marks = re.compile(
            r"[^A-Za-z\d\u3005\u3040-\u30fa\u30fc-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
        )

        # List of (symbol, Japanese) pairs for marks:
        self._symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]

    def symbols_to_japanese(self, text):
        for regex, replacement in self._symbols_to_japanese:
            text = re.sub(regex, replacement, text)
        return text
    
    # Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    def _numeric_feature_by_regex(self, regex, s):
        match = re.search(regex, s)
        if match is None:
            return -50
        return int(match.group(1))
    
    def pyopenjtalk_g2p_prosody(self, text, word2ph, drop_unvoiced_vowels=True):
        features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(features)
        N = len(labels)

        phones = []
        node_phone_counts = [0] * len(features)

        expected_base_counts = []
        for node in features:
            if node['pron'] == 'IDLE':
                expected_base_counts.append(0)
            else:
                # 使用 g2p 获取该词的标准音素序列
                # 用 split() 打断后统计长度，这就是该词应占用的基础音素数量
                with suppress_c_stderr():
                    ph_str = pyopenjtalk.g2p(node['pron'])
                expected_base_counts.append(len(ph_str.split()) if ph_str else 0)

        node_idx = 0
        current_base_consumed = 0

        # 跳过开头那些发音为空（如 IDLE）的节点
        while node_idx < len(features) - 1 and expected_base_counts[node_idx] == 0:
            node_idx += 1

        for n in range(N):
            lab_curr = labels[n]
            p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
            
            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()

            has_other = False
            if p3 not in ["sil", "pau"]:
                a1 = self._numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
                a2 = self._numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
                a3 = self._numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
                f1 = self._numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
                a2_next = self._numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1]) if n+1 < N else -1

                if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                    phones.append("#")
                    has_other = True
                elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                    phones.append("]")
                    has_other = True
                elif a2 == 1 and a2_next == 2:
                    phones.append("[")
                    has_other = True
                
                if has_other:
                    node_phone_counts[node_idx] += 1

            res_p = None
            is_sentence_boundary_sil = False

            if p3 == "sil":
                if n == 0: 
                    res_p = "^"
                    is_sentence_boundary_sil = True # 句首静音不计入词的发音配额
                elif n == N - 1:
                    e3 = self._numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    res_p = "$" if e3 == 0 else "?"
                    is_sentence_boundary_sil = True # 句尾静音不计入词的发音配额
                else:
                    res_p = "_"
            elif p3 == "pau":
                res_p = "_"
            else:
                res_p = p3

            if res_p:
                phones.append(res_p)
                
                # 如果是句首句尾将要被裁掉的 sil，就不算进词汇的音素数量里
                if not is_sentence_boundary_sil:
                    node_phone_counts[node_idx] += 1
                    current_base_consumed += 1

                    # 如果当前词的配额已经用完，且还没到句子末尾，就移动到下一个词
                    while node_idx < len(features) - 1 and current_base_consumed >= expected_base_counts[node_idx]:
                        current_base_consumed -= expected_base_counts[node_idx] # 清空已用配额
                        node_idx += 1 # 指针推向下个词

        for i, node in enumerate(features):
            surface = node['string']
            if node['pron'] == 'IDLE': continue
            
            total_ph_count = node_phone_counts[i]
            num_chars = len(surface)
            
            if num_chars <= 1:
                word2ph["word"].append(surface)
                word2ph["ph"].append(total_ph_count)
            else:
                # 由于在日语中，一个字对应的音素长度是不固定的，所以这里直接按字符数平分
                avg_ph = total_ph_count // num_chars
                remainder = total_ph_count % num_chars
                for j in range(num_chars):
                    word2ph["word"].append(surface[j])
                    word2ph["ph"].append(avg_ph + 1 if j < remainder else avg_ph)

        return phones, word2ph
    
    def preprocess_jap(self, text, with_prosody=False):
        """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
        text = self.symbols_to_japanese(text)
        # English words to lower case, should have no influence on japanese words.
        text = text.lower()
        sentences = re.split(self._japanese_marks, text)
        marks = re.findall(self._japanese_marks, text)

        text = []
        word2ph = {"word":[], "ph":[]}
        for i, sentence in enumerate(sentences):
            if re.match(self._japanese_characters, sentence):
                if with_prosody:
                    ph, word2ph = self.pyopenjtalk_g2p_prosody(sentence, word2ph)
                    text += ph[1:-1]
                else:
                    p = pyopenjtalk.g2p(sentence)
                    text += p.split(" ")

            if i < len(marks):
                if marks[i] == " ":  # 防止意外的UNK
                    continue
                text += [marks[i].replace(" ", "")]
                word2ph["word"].append(marks[i])
                word2ph["ph"].append(1)
                
        return text, word2ph

    def g2p(self, norm_text, with_prosody=True):
        phones, word2ph = self.preprocess_jap(norm_text, with_prosody)
        return phones, word2ph