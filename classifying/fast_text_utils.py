from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import utils.pattern_utils as pu
import fasttext_pybind as fasttext


print('my fast')
loss_name = fasttext.loss_name
model_name = fasttext.model_name
EOS, BOW, EOW = "</s>", "<", ">"


class FastText:
    """
    This class defines the API to inspect models and should not be used to
    create objects. It will be returned by functions such as load_model or
    train.

    In general this API assumes to be given only unicode for Python2 and the
    Python3 equvalent called str for any string-like arguments. All unicode
    strings are then encoded as UTF-8 and fed to the fastText C++ API.
    """
    
    def __init__(self, model=None):
        self.ftobj = fasttext.fasttext()
        if model is not None:
            self.ftobj.loadModel(model)
    
    def is_quantized(self):
        return self.ftobj.isQuant()
    
    def get_new_vector(self):
        return fasttext.Vector(self.get_dimension())
    
    def get_dimension(self):
        """Get the dimension of word vector """
        return self.ftobj.getArgs().dim
    
    def get_word_vector(self, word):
        """Get the vector of a word."""
        v = self.get_new_vector()
        self.ftobj.getWordVector(v, word)
        return np.array(v)
    
    def get_sentence_vector(self, text):
        """
        Given a string, get a single vector represenation. This function
        assumes to be given a single line of text. We split words on
        whitespace (space, newline, tab, vertical tab) and the control
        characters carriage return, formfeed and the null character.
        """
        if text.find('\n') != -1:
            raise ValueError("predict processes one line at a time (remove \'\\n\')")
        text += "\n"
        v = self.get_new_vector()
        self.ftobj.getSentenceVector(v, text)
        return np.array(v)
    
    def get_word_id(self, word):
        """ Return the word id if in the dictionary else -1. """
        return self.ftobj.getWordId(word)
    
    def get_subword_id(self, subword):
        """ Given a subword, return the index (within input matrix) it hashes to. """
        return self.ftobj.getSubwordId(subword)
    
    def get_subwords(self, word):
        """ Given a word, get the subwords and their indicies. """
        pair = self.ftobj.getSubwords(word)
        return pair[0], np.array(pair[1])
    
    def get_input_vector(self, ind):
        """ Given an index, get the corresponding vector of the Input Matrix. """
        v = self.get_new_vector()
        self.ftobj.getInputVector(v, ind)
        return np.array(v)
    
    def predict(self, text, k=1, threshold=0.5):
        """
        Given a string, get a list of labels and a list of
        corresponding probabilities. k controls the number
        of returned labels. A choice of 5, will return the 5
        most probable labels. By default this returns only
        the most likely label and probability. threshold filters
        the returned labels by a threshold on probability. A
        choice of 0.5 will return labels with at least 0.5
        probability. k and threshold will be applied together to
        determine the returned labels.

        This function assumes to be given
        a single line of text. We split words on whitespace (space,
        newline, tab, vertical tab) and the control characters carriage
        return, formfeed and the null character.

        If the model is not supervised, this function will throw a ValueError.

        If given a list of strings, it will return a list of results as usually
        received for a single line of text.
        """
        if type(text) == list:
            text = [check(entry) for entry in text]
            all_probs, all_labels = self.ftobj.multilinePredict(text, k, threshold)
            return all_labels, np.array(all_probs, copy=False)
        else:
            text = check(text)
            pairs = self.ftobj.predict_proba(text, k, threshold)
            probs, labels = zip(*pairs)
            return labels, np.array(probs, copy=False)
    
    def get_input_matrix(self):
        """
        Get a copy of the full input matrix of a Model. This only
        works if the model is not quantized.
        """
        if self.ftobj.isQuant():
            raise ValueError("Can't get quantized Matrix")
        return np.array(self.ftobj.getInputMatrix())
    
    def get_output_matrix(self):
        """
        Get a copy of the full output matrix of a Model. This only
        works if the model is not quantized.
        """
        if self.ftobj.isQuant():
            raise ValueError("Can't get quantized Matrix")
        return np.array(self.ftobj.getOutputMatrix())
    
    def get_words(self, include_freq=False):
        """
        Get the entire list of words of the dictionary optionally
        including the frequency of the individual words. This
        does not include any subwords. For that please consult
        the function get_subwords.
        """
        pair = self.ftobj.getVocab()
        if include_freq:
            return pair[0], np.array(pair[1])
        else:
            return pair[0]
    
    def get_labels(self, include_freq=False):
        """
        Get the entire list of labels of the dictionary optionally
        including the frequency of the individual labels. Unsupervised
        models use words as labels, which is why get_labels
        will call and return get_words for this type of
        model.
        """
        a = self.ftobj.getArgs()
        if a.model == model_name.supervised:
            pair = self.ftobj.getLabels()
            if include_freq:
                return pair[0], np.array(pair[1])
            else:
                return pair[0]
        else:
            return self.get_words(include_freq)
    
    def get_line(self, text):
        """
        Split a line of text into words and labels. Labels must start with
        the prefix used to create the model (__label__ by default).
        """
        if type(text) == list:
            text = [check(entry) for entry in text]
            return self.ftobj.multilineGetLine(text)
        else:
            text = check(text)
            return self.ftobj.getLine(text)
    
    def test(self, path, k=1):
        """ Evaluate supervised model using file given by path """
        return self.ftobj.test(path, k)
    
    def quantize(
            self,
            input=None,
            qout=False,
            cutoff=0,
            retrain=False,
            epoch=None,
            lr=None,
            thread=None,
            verbose=None,
            dsub=2,
            qnorm=False
    ):
        """ Quantize the model reducing the size of the model and it's memory footprint. """
        a = self.ftobj.getArgs()
        if not epoch:
            epoch = a.epoch
        if not lr:
            lr = a.lr
        if not thread:
            thread = a.thread
        if not verbose:
            verbose = a.verbose
        if retrain and not input:
            raise ValueError("Need input file path if retraining")
        if input is None:
            input = ""
        self.ftobj.quantize(input, qout, cutoff, retrain, epoch, lr, thread, verbose, dsub, qnorm)
    
    def train_supervised(
            self,
            input,
            lr=0.1,
            dim=100,
            ws=5,
            epoch=5,
            minCount=1,
            minCountLabel=0,
            minn=0,
            maxn=0,
            neg=5,
            wordNgrams=1,
            loss="softmax",
            bucket=2000000,
            thread=12,
            lrUpdateRate=100,
            t=1e-4,
            label="__label__",
            verbose=2,
            pretrainedVectors=""):
        """
        Train a supervised model and return a model object.

        input must be a filepath. The input text does not need to be tokenized
        as per the tokenize function, but it must be preprocessed and encoded
        as UTF-8. You might want to consult standard preprocessing scripts such
        as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html

        The input file must must contain at least one label per line. For an
        example consult the example datasets which are part of the fastText
        repository such as the dataset pulled by classification-example.sh.
        """
        model = "supervised"
        args = locals()
        args.pop('self')
        a = _build_args(args)
        fasttext.train(self.ftobj, a)
    
    def train_unsupervised(
            self,
            input,
            model="skipgram",
            lr=0.05,
            dim=100,
            ws=5,
            epoch=5,
            minCount=5,
            minCountLabel=0,
            minn=3,
            maxn=6,
            neg=5,
            wordNgrams=1,
            loss="ns",
            bucket=2000000,
            thread=12,
            lrUpdateRate=100,
            t=1e-4,
            label="__label__",
            verbose=2,
            pretrainedVectors=""):
        """
        Train an unsupervised model and return a model object.

        input must be a filepath. The input text does not need to be tokenized
        as per the tokenize function, but it must be preprocessed and encoded
        as UTF-8. You might want to consult standard preprocessing scripts such
        as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html

        The input field must not contain any labels or use the specified label prefix
        unless it is ok for those words to be ignored. For an example consult the
        dataset pulled by the example script word-vector-example.sh, which is
        part of the fastText repository.
        """
        args = locals()
        args.pop('self')
        a = _build_args(args)
        fasttext.train(self.ftobj, a)


def check(entry):
    if entry.find('\n') != -1:
        raise ValueError("should processes one line at a time (remove \'\\n\')")
    return entry + '\n'


def _parse_model_string(string):
    model_name_dict = {'cbow': model_name.cbow, 'skipgram': model_name.skipgram, 'supervised': model_name.supervised}
    return model_name_dict[string]


def _parse_loss_string(string):
    loss_name_dict = {'ns': loss_name.ns, 'hs': loss_name.hs, 'softmax': loss_name.softmax}
    return loss_name_dict[string]


def _build_args(args):
    args["model"] = _parse_model_string(args["model"])
    args["loss"] = _parse_loss_string(args["loss"])
    a = fasttext.args()
    for (k, v) in args.items():
        setattr(a, k, v)
    a.output = ""  # User should use save_model
    a.pretrainedVectors = ""  # Unsupported
    a.saveOutput = 0  # Never use this
    if a.wordNgrams <= 1 and a.maxn == 0:
        a.bucket = 0
    return a


ftobj = fasttext.fasttext()


def tokenize(text):
    """Given a string of text, tokenize it and return a list of tokens"""
    return ftobj.tokenize(text)


""" my definitions """
model_dict = dict()
label_t, value_t = '__label__true', 1
label_f, value_f = '__label__false', 0
binary_label2value = {label_t: value_t, label_f: value_f}


def load_model(path):
    return FastText(path)


def save_model(path, model):
    model.ftobj.saveModel(path)


def get_model(model_file):
    """ restores the models that has already been loaded into memory """
    if model_file not in model_dict:
        print('model load over ', type(model_file))
        model_dict[model_file] = load_model(model_file)
    return model_dict[model_file]


def train_supervised(model_file, *args, **kwargs):
    """ currently no online update, the model will be saved to the file if model_file is provided """
    model = FastText()
    model.train_supervised(*args, **kwargs)
    if model_file is not None:
        save_model(model_file, model)
    else:
        print('model not saved')
    return model


def binary_predict(target, model, threshold):
    if type(target) is str:
        # single text classification
        text = target.strip()
        if pu.is_empty_string(text):
            return value_f
        else:
            pred, score = model.predict_proba(text, threshold=threshold)
            return binary_label2value[pred[0]], score[0]
    else:
        # multi-line text classification
        text_arr, ignore_idx_arr, pred_value_arr = list(), list(), list()
        for idx, text in enumerate(target):
            assert type(text) is str
            text = text.strip()
            if pu.is_empty_string(text):
                ignore_idx_arr.append(idx)
            else:
                text_arr.append(text)
        pred_label_arr, score_arr = model.predict_proba(text_arr, threshold=threshold)
        assert len(pred_label_arr) == len(text_arr) and len(score_arr) == len(text_arr)
        for idx in range(len(text_arr)):
            pred = pred_label_arr[idx]
            # print(score_arr[idx], pred)
            if len(pred) == 0:
                # this text has been ignored due to some unidentifiable reason
                value = binary_predict(text_arr[idx], model, threshold)
            else:
                value = binary_label2value[pred[0]]
            pred_value_arr.append(value)
        for idx in ignore_idx_arr:
            pred_value_arr.insert(idx, value_f)
        score_arr = [s[0] for s in score_arr]
        return pred_value_arr, score_arr


def predict(target, model, threshold=0.5):
    """ returns value/value array given input text/text array; value(s) are dependent on the threshold """
    pred_value_arr, score_arr = binary_predict(target, model, threshold)
    return pred_value_arr, score_arr

