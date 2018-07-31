import os
import re
import utils.nltk_utils as nu
from wordsegment import load, segment


file_path = os.path.abspath(os.path.dirname(__file__))


def load_stop_words():
    stop_words_file = os.path.join(file_path, "stopwords.txt")
    with open(stop_words_file) as fp:
        lines = fp.readlines()
    words = set([line.strip() for line in lines])
    return words


def load_abbr_words():
    abbr_words_file = os.path.join(file_path, "abbrs.txt")
    with open(abbr_words_file) as fp:
        lines = fp.readlines()
    words = set([tuple(line.strip().split('\t')) for line in lines])
    return words


def load_emoticon_words():
    emot_words_file = os.path.join(file_path, "emoticons.txt")
    with open(emot_words_file) as fp:
        lines = fp.readlines()
    words = set([line.strip() for line in lines])
    return words


load()
nltk_stop_words = nu.NltkFactory.load_stop_words()
my_stop_words = load_stop_words()
abbr_words = load_abbr_words()
emot_words = load_emoticon_words()


class PatternHolder:
    def __init__(self, exp_sub_list):
        self.exp_list = list()
        self.update_pattern_dict(exp_sub_list)
    
    def update_pattern_dict(self, exp_sub_list):
        for exp_sub in exp_sub_list:
            if len(exp_sub) == 3:
                exp, sub, cap_ignore = exp_sub
            elif len(exp_sub) == 2:
                (exp, sub), cap_ignore = exp_sub, True
            else:
                raise ValueError('wrong format ' + str(exp_sub))
            flags = re.I if cap_ignore else 0
            self.exp_list.append([exp, re.compile(exp, flags=flags), sub])
    
    def apply_patterns(self, text):
        for exp, pattern, sub in self.exp_list:
            text = pattern.sub(sub, text)
        return text


tw_rule_list = [(r'RT\s@?.*?:(\s|$)', ' '), (r'@\w+', ' '), (r'(#.+?)(\s|$)', ' \\1 '),
                (r'https?:\W*/.*?(\s|$)|https?:(\s|$)', ' '), ]
special_char_list = [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                     ('&ndash;', ' '), ('&mdash;', ' '), ('[’‘]', '\''), ]
puctuation_list = [(r'[~`$^()\-_=\+\[\]{}"|:;]', ' '), ]
word_rule_list = [('^[^A-Za-z0-9]+$', ''), ('^\W*', ''), ('\W*$', ''), ("'[sS]?\W*$", ''),
                  ('(.+?)\\1{3,}', '\\1'), ('^(.+?)\\1{3,}$', ''), ]
contraction_list = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'),
                    (r'ain[\'\s]t', 'are not'), (r'hasn\'t', 'has not'), (r'isn\'t', 'is not'),
                    (r'(\W|^)you[\'\s]?re(\W|$)', '\\1you are\\2'), (r'(\w+)n\'t', '\g<1> not'),
                    (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)\'d', '\g<1> would'),
                    (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'re', '\g<1> are'), ]

tw_rule_patterns = PatternHolder(tw_rule_list)
special_patterns = PatternHolder(special_char_list)
word_patterns = PatternHolder(word_rule_list)
contraction_patterns = PatternHolder(contraction_list)
punc_patterns = PatternHolder(puctuation_list)

endline_pattern = PatternHolder([(r'[\n\r]+$', ''), ])
brkline_pattern = PatternHolder([(r'[\n\r]+', '.'), ])
dupspace_pattern = PatternHolder([(r'\s\s+', ' '), ])
nonascii_pattern = PatternHolder([(r'[^\x00-\x7f]', '.'), ])

abbr_list = [w for w in abbr_words]
abbr_patterns = PatternHolder(abbr_list)


def emoticon_normalization(text):
    for emot_word in emot_words:
        text = text.replace(emot_word, ' ')
    return text


# r"[a-zA-Z0-9]+(?:['_-][a-zA-Z0-9]+)*"
tokenize_pattern = r"[a-zA-Z0-9]+(?:['_-][a-zA-Z0-9]+)*"


def findall(pattern, string, flags=re.I): return re.findall(pattern, string, flags)


def search_pattern(pattern, string, flags=re.I): return re.search(pattern, string, flags)


def sub_pattern(pattern, repl, string, flags=re.I): return re.sub(pattern, repl, string, flags)


def remove_non_ascii(text): return nonascii_pattern.apply_patterns(text)


def remove_duplicate_spaces(text): return dupspace_pattern.apply_patterns(text)


def remove_endline(text): return endline_pattern.apply_patterns(text)


def remove_breakline(text): return brkline_pattern.apply_patterns(remove_endline(text))


def split_digit_arr(text): return re.findall('\d+', text)


def has_azAZ(text): return re.search(r'[a-zA-Z]+', text.strip()) is not None


def is_empty_string(text): return len(text) == 0 or len(re.findall(r'^\s+$', text)) > 0


def is_stop_word(text, stopwords_corpus): return text.strip().lower() in stopwords_corpus


def is_char(text): return len(text) == 1


def has_enough_alpha(text, threshold):
    text = re.sub('\s', '', text)
    if len(text) == 0:
        return False
    alphas = re.findall('[a-zA-Z]', text)
    return len(alphas) / len(text) >= threshold


def text_normalization(text):
    pattern_list = [special_patterns, nonascii_pattern, endline_pattern, brkline_pattern,
                    tw_rule_patterns, contraction_patterns, dupspace_pattern, ]
    for pattern in pattern_list:
        text = pattern.apply_patterns(text)
    return text


def temporal_text_normalization(text):
    pattern_list = [nonascii_pattern, endline_pattern, brkline_pattern,
                    tw_rule_patterns, contraction_patterns, dupspace_pattern, ]
    for pattern in pattern_list:
        text = pattern.apply_patterns(text)
    return text


def capitalize(phrase):
    words = [w for w in re.split('\W', phrase) if not is_empty_string(w)]
    capitalize = [w.capitalize() for w in words]
    return ' '.join(capitalize)


def is_valid_keyword(word):
    if not word:
        return False
    # haschar = re.search('^[^a-zA-Z#]', word) is None
    haschar = has_azAZ(word)
    notsinglechar = len(word) > 1
    notstopword = not is_stop_word(word, my_stop_words)
    return haschar and notsinglechar and notstopword


def tokenize(text): return findall(tokenize_pattern, text)


def valid_tokenize(text):
    return [t.strip() for t in findall(tokenize_pattern, text) if is_valid_keyword(t)]


if __name__ == '__main__':
    # print(emoticon_list)
    # print(abbr_list)
    print(emoticon_normalization('-_-,=_=qwuerT_T)&(*ESPNUIYNWA'))
    # s = 'Bahrain bans all protests in new crackdown. http://t.co/l0AQNtmB'
    # ss = re.sub(r'https?:\W*/.*?(\s|$)|https?:(\s|$)', ' ', s)
    # print(ss)
    print(text_normalization('Bahrain bans all protests in new crackdown. http://t.co/l0AQNtmB'))
