
class NltkFactory:
    lemm = stem = stop = None
    
    @staticmethod
    def lemmatize(word):
        if not NltkFactory.lemm:
            from nltk.stem import WordNetLemmatizer
            NltkFactory.lemm = WordNetLemmatizer()
        return NltkFactory.lemm.lemmatize(word)
    
    @staticmethod
    def stemming(word):
        if not NltkFactory.stem:
            from nltk.stem.snowball import EnglishStemmer
            NltkFactory.stem = EnglishStemmer()
        return NltkFactory.stem.stem(word)
    
    @staticmethod
    def load_stop_words():
        if not NltkFactory.stop:
            from nltk.corpus import stopwords
            NltkFactory.stop = set(stopwords.words('english'))
        return NltkFactory.stop


lemmatize = NltkFactory.lemmatize
stemming = NltkFactory.stemming
