import utils.tweet_keys as tk
from collections import Counter
import re


NOT_NUM = 'no'


class Level:
    def __init__(self):
        self.injured_word = {'seriously', 'wounded', 'injured', 'critical', 'hurt', 'injuring', 'casualties',
                             'moderately', 'wounding', 'wounds', 'civilians', 'wound', 'missing', 'least',
                             'people'}
        self.death_word = {'kill', 'kills', 'killed', 'killing', 'dead', 'death', 'martyrs'}
    
    def get_level(self, twarr):
        death_num, injured_num = self.get_max_death_injure(twarr)
        count = death_num + injured_num
        if count < 30:
            level = '一般事件'
        elif 30 <= count < 100:
            level = '较大事件'
        elif 100 <= count < 300:
            level = '重大事件'
        else:
            level = '特别重大事件'
        return level
    
    def get_max_death_injure(self, twarr):
        death_dict, injured_dict = Counter(), Counter()
        textarr = [tw.get(tk.key_text) for tw in twarr]
        for text in textarr:
            try:
                death_num, injured_num = self.get_single_tweet_injured_death(text)
            except:
                death_num = injured_num = 0
            if injured_num > 0:
                injured_dict[injured_num] += 1
            if death_num > 0:
                death_dict[death_num] += 1
        death_number = self.get_max_key_with_max_count(death_dict.most_common())
        injured_number = self.get_max_key_with_max_count(injured_dict.most_common())
        return death_number, injured_number
    
    def get_max_key_with_max_count(self, key_count_list):
        # print(key_count_list)
        if not key_count_list:
            return 0
        max_key, max_cnt = key_count_list[0]
        for key, cnt in key_count_list[1:]:
            if cnt < max_cnt:
                break
            else:
                if key > max_key:
                    max_key = key
        return max_key
    
    def get_single_tweet_injured_death(self, text):
        word_list = text.lower().split()
        injured_number = death_number = 0
        for i in range(1, len(word_list) - 1):
            if word_list[i] in self.injured_word:
                prev_n = self.spoken_word_to_number(word_list[i - 1])
                next_n = self.spoken_word_to_number(word_list[i + 1])
                if type(prev_n) is int and prev_n > 0:
                    injured_number = prev_n
                elif type(next_n) is int and next_n > 0:
                    injured_number = next_n
            if word_list[i] in self.death_word:
                prev_n = self.spoken_word_to_number(word_list[i - 1])
                next_n = self.spoken_word_to_number(word_list[i + 1])
                if type(prev_n) is int and prev_n > 0:
                    death_number = prev_n
                elif type(next_n) is int and next_n > 0:
                    death_number = next_n
        return death_number, injured_number
    
    def spoken_word_to_number(self, w):
        w = w.lower().strip()
        number = re.findall('\d+', w)
        if number:
            return int(number[0])
        _known = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
            'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
            'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        if w in _known:
            return _known[w]
        else:
            inputWordArr = re.split('-', w)
        if len(inputWordArr) <= 1:
            return NOT_NUM
        # Check the pathological case where hundred is at the end or thousand is at end
        if inputWordArr[-1] == 'hundred':
            inputWordArr.append('zero')
            inputWordArr.append('zero')
        if inputWordArr[-1] == 'thousand':
            inputWordArr.append('zero')
            inputWordArr.append('zero')
            inputWordArr.append('zero')
        if inputWordArr[0] == 'hundred':
            inputWordArr.insert(0, 'one')
        if inputWordArr[0] == 'thousand':
            inputWordArr.insert(0, 'one')
        inputWordArr = [word for word in inputWordArr if word not in {'and', 'minus', 'negative'}]
        for word in inputWordArr:
            if word not in _known:
                return NOT_NUM
                # return 0
        output = 0
        currentPosition = 'unit'
        for word in reversed(inputWordArr):
            if currentPosition == 'unit':
                number = _known[word]
                output += number
                if number > 9:
                    currentPosition = 'hundred'
                else:
                    currentPosition = 'ten'
            elif currentPosition == 'ten':
                if word != 'hundred':
                    number = _known[word]
                    if number < 10:
                        output += number * 10
                    else:
                        output += number
                currentPosition = 'hundred'
            elif currentPosition == 'hundred':
                if word not in ['hundred', 'thousand']:
                    number = _known[word]
                    output += number * 100
                    currentPosition = 'thousand'
                elif word == 'thousand':
                    currentPosition = 'thousand'
                else:
                    currentPosition = 'hundred'
            elif currentPosition == 'thousand':
                if word != 'thousand':
                    number = _known[word]
                    output += number * 1000
        return output


class AttackLevel(Level):
    def __int__(self):
        Level.__init__(self)


if __name__ == '__main__':
    import utils.file_iterator as fi
    import utils.function_utils as fu
    from pathlib import Path
    base = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive"
    files = fi.listchildren(base, concat=True)
    h = Level()
    for f in files:
        _twarr = fu.load_array(f)
        print(Path(f).name, h.get_level(_twarr), '\n\n')
