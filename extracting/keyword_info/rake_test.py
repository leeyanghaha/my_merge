# from __future__ import absolute_import
# from __future__ import print_function
import re
import six
import operator
from six.moves import range
from collections import Counter

import utils.pattern_utils as pu


debug = True
test = False
tokenize = re.compile('[^a-zA-Z0-9_\\+\\-/]')
sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')


def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def split_words(text, min_word_return_size):
    """ Returns a list of words where each word length > min_word_return_size. """
    words = list()
    for word in tokenize.split(text):
        word = word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(word) > min_word_return_size and len(word) > 0 and not is_number(word):
            words.append(word)
    return words


def split_sentences(text):
    return sentence_delimiters.split(text)


def load_stop_words(stop_word_file):
    stop_words = list()
    for line in open(stop_word_file).readlines():
        line = line.strip()
        if line[0:1] == "#":
            continue
        stop_words.extend(line.split())
    return stop_words


def build_stop_word_pattern(stop_words):
    stop_word_regex_list = list()
    for word in stop_words:
        word_regex = '\\b{}\\b'.format(word)
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


# Function that extracts the adjoined candidates from a list of sentences and filters them by frequency
def extract_adjoined_candidates(sentence_list, stoplist, min_keywords, max_keywords, min_freq):
    adjoined_candidates = list()
    for s in sentence_list:
        # Extracts the candidates from each single sentence and adds them to the list
        adjoined_candidates += adjoined_candidates_from_sentence(s, stoplist, min_keywords, max_keywords)
    # Filters the candidates and returns them
    return filter_adjoined_candidates(adjoined_candidates, min_freq)


# Function that extracts the adjoined candidates from a single sentence
def adjoined_candidates_from_sentence(s, stop_word_list, min_keyword_num, max_keyword_num):
    candidates = list()
    words = s.lower().split()
    # For each possible length of the adjoined candidate
    for keyword_num in range(min_keyword_num, max_keyword_num + 1):
        for i in range(0, len(words) - keyword_num):
            # Proceeds only if the first word of the candidate is not a stopword
            if words[i] not in stop_word_list:
                candidate = words[i]
                # Initializes j (the pointer to the next word) to 1
                j = 1
                # Initializes the word counter. This counts the non-stopwords words in the candidate
                keyword_counter = 1
                contains_stopword = False
                # Until the word count reaches the maximum number of keywords or the end is reached
                while keyword_counter < keyword_num and i + j < len(words):
                    # Adds the next word to the candidate
                    candidate = candidate + ' ' + words[i + j]
                    # If it's not a stopword, increase the word counter. If it is, turn on the flag
                    if words[i + j] not in stop_word_list:
                        keyword_counter += 1
                    else:
                        contains_stopword = True
                    # Next position
                    j += 1
                # Adds the candidate to the list only if:
                # 1) it contains at least a stopword (if it doesn't it's already been considered)
                # AND
                # 2) the last word is not a stopword
                # AND
                # 3) the adjoined candidate keyphrase contains exactly the correct number of keywords (to avoid doubles)
                if contains_stopword and candidate.split()[-1] not in stop_word_list and keyword_counter == keyword_num:
                    candidates.append(candidate)
    return candidates


# Function that filters the adjoined candidates to keep only those that appears with a certain frequency
def filter_adjoined_candidates(candidates, min_freq):
    # Creates a dictionary where the key is the candidate and the value is the frequency of the candidate
    candidates_freq = Counter(candidates)
    filtered_candidates = []
    # Uses the dictionary to filter the candidates
    for candidate in candidates:
        freq = candidates_freq[candidate]
        if freq >= min_freq:
            filtered_candidates.append(candidate)
    return filtered_candidates


def generate_candidate_keywords(
        sentence_list, stopword_pattern, stop_word_list, min_char_num=1, max_words_num=5,
        min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2):
    phrase_list = list()
    adjoined_candidates = list()
    for s in sentence_list:
        replace_stop = re.sub(stopword_pattern, "|", s.strip())
        phrases = replace_stop.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if not pu.is_empty_string(phrase) and is_acceptable_phrase(phrase, min_char_num, max_words_num):
                phrase_list.append(phrase)
        
        adjoined_candidates += adjoined_candidates_from_sentence(
            s, stop_word_list, min_words_length_adj, max_words_length_adj)
    phrase_list += filter_adjoined_candidates(adjoined_candidates, min_phrase_freq_adj)
    
    phrase_list += extract_adjoined_candidates(
        sentence_list, stop_word_list, min_words_length_adj, max_words_length_adj, min_phrase_freq_adj)
    return phrase_list


def is_acceptable_phrase(phrase, min_char_num, max_words_num):
    if len(phrase) < min_char_num:
        return False
    words = phrase.split()
    if len(words) > max_words_num:
        return False
    digit_num = alpha_num = 0
    for c in phrase:
        if c.isdigit():
            digit_num += 1
        elif c.isalpha():
            alpha_num += 1
    if alpha_num == 0 or digit_num > alpha_num:
        return False
    return True


def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = split_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        # if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  # orig.
            # word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
    # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score


def generate_candidate_keyword_scores(phrase_list, word_score, min_keyword_frequency=1):
    keyword_candidates = {}
    for phrase in phrase_list:
        if min_keyword_frequency > 1:
            if phrase_list.count(phrase) < min_keyword_frequency:
                continue
        keyword_candidates.setdefault(phrase, 0)
        word_list = split_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates


class Rake(object):
    def __init__(self, stop_words_path, min_char_length=1, max_words_length=5, min_keyword_frequency=1,
                 min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=2):
        self.__stop_words_path = stop_words_path
        self.__stop_words_list = load_stop_words(stop_words_path)
        self.__min_char_length = min_char_length
        self.__max_words_length = max_words_length
        self.__min_keyword_frequency = min_keyword_frequency
        self.__min_words_length_adj = min_words_length_adj
        self.__max_words_length_adj = max_words_length_adj
        self.__min_phrase_freq_adj = min_phrase_freq_adj

    def run(self, text):
        sentence_list = split_sentences(text)

        stop_words_pattern = build_stop_word_pattern(self.__stop_words_list)

        phrase_list = generate_candidate_keywords(sentence_list, stop_words_pattern, self.__stop_words_list,
                                                  self.__min_char_length, self.__max_words_length,
                                                  self.__min_words_length_adj, self.__max_words_length_adj,
                                                  self.__min_phrase_freq_adj)

        word_scores = calculate_word_scores(phrase_list)

        keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores, self.__min_keyword_frequency)

        sorted_keywords = sorted(six.iteritems(keyword_candidates), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords


if test and __name__ == '__main__':
    text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
    sentenceList = split_sentences(text)
    # stoppath = "data/stoplists/SmartStoplist.txt"
    stoppath = "/home/nfs/cdong/tw/src/tools/AutoPhrase/data/EN/stopwords.txt"
    stopwordpattern = build_stop_word_pattern(stoppath)
    phraseList = generate_candidate_keywords(sentenceList, stopwordpattern, load_stop_words(stoppath))

    # calculate individual word scores
    wordscores = calculate_word_scores(phraseList)

    # generate candidate keyword_info scores
    keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)
    if debug: print(keywordcandidates)
    sortedKeywords = sorted(six.iteritems(keywordcandidates), key=operator.itemgetter(1), reverse=True)
    totalKeywords = len(sortedKeywords)
    if debug: print(sortedKeywords)
    if debug: print(totalKeywords)
    if debug: print(sortedKeywords[0:(totalKeywords // 3)])

    rake = Rake("data/stoplists/SmartStoplist.txt")
    keywords = rake.run(text)
    print(keywords)
