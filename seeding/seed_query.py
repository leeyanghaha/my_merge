import os
import re

from utils.synset import get_synset
import utils.date_utils as du
import utils.pattern_utils as pu


class SeedQuery:
    def __init__(self, keywords, since, until):
        self.all = self.any = self.none = None
        self.keywords = keywords if keywords is not None else {}
        self.since = since if since is not None else []
        self.until = until if since is not None else []
        self.month_dict = {'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4', 'May': '5', 'June': '6',
                           'July': '7', 'Aug': '8', 'Sept': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
        self.query_results = list()
        self.parse_keywords(keywords)
    
    def parse_keywords(self, keywords):
        self.all = keywords['all_of'] if 'all_of' in keywords else []
        self.any = keywords['any_of'] if 'any_of' in keywords else []
        self.none = keywords['none_of'] if 'none_of' in keywords else []
    
    def expand_keywords(self, keywords):
        if not keywords:
            return keywords
        mylabel = r'\W'
        for idx in range(len(keywords) - 1, -1, -1):
            w_synset = get_synset(keywords[idx].strip(mylabel))
            if keywords[idx].startswith(mylabel):
                w_synset = [mylabel + w for w in w_synset]
            if keywords[idx].endswith(mylabel):
                w_synset = [w + mylabel for w in w_synset]
            keywords.extend(w_synset)
            del keywords[idx]
        return keywords
    
    def append_desired_tweet(self, tw, usingtwtime=False):
        """
        Every time a tw is entered, absorb tw if it satisfies query(keywords, since, until) of this query.
        :param tw: Tweet of type Dict(), key 'text' is necessary.
        :param usingtwtime: If use tw['created_at'] as time of tweet.
        :return: If the tweet has been accepted by this query.
        """
        if usingtwtime and not self.is_time_desired(self.time_of_tweet(tw['created_at'], 'tweet')):
            return False
        if not self.is_text_desired(tw['text']):
            return False
        self.query_results.append(tw)
        return True
    
    def time_of_tweet(self, time_about_str, source='filename'):
        if source is 'filename':
            file_name = os.path.basename(time_about_str)
            time_str = file_name[:-4]
            return pu.split_digit_arr(time_str)[0:3]
        if source is 'tweet':
            time_split = time_about_str.split(' ')
            return [time_split[-1], self.month_dict[time_split[1]], time_split[2]]
    
    def is_time_desired(self, tw_ymd):
        return du.is_target_ymdh(tw_ymd, self.since, self.until)
    
    def is_text_desired(self, text):
        if (self.all is []) and (self.any is []) and (self.none is []):
            return True
        for w1 in self.all:
            if not re.search(w1, text, re.IGNORECASE):
                return False
        for w2 in self.none:
            if re.search(w2, text, re.IGNORECASE):
                return False
        if not self.any:
            return True
        # complementary pass, then additional
        for w3 in self.any:
            if re.search(w3, text, re.IGNORECASE):
                return True
        return False
    
    def to_string(self):
        return '_'.join([
            '-'.join(["{:0>2}".format(t) for t in self.since]),
            '-'.join(self.any[0].strip('\W').split(' ')),
            '-'.join(self.all[0].strip('\W').split(' ')),
        ])
