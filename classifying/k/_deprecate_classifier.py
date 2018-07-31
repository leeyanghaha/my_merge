from config.configure import config
import classifying.fast_text_utils as ftu

import utils.array_utils as au


label2value = ftu.binary_label2value
korea_model_file = config.korea_ft_model_file


class ClassifyK:
    def __init__(self, ft_model_file):
        self.ft_model = None
        if ft_model_file is not None:
            self.load_fasttext_model(ft_model_file)
        
    def load_fasttext_model(self, ft_model_file):
        self.ft_model = ftu.load_model(ft_model_file)


def predict(textarr, threshold=0.5):
    """ returns value/value array given input text/text array; value(s) are dependent on the threshold """
    model = ftu.get_model(korea_model_file)
    pred_value_arr = ftu.binary_predict(textarr, model, threshold)
    return pred_value_arr


def train(train_file, model_file):
    model = ftu.FastText()
    model.train_supervised(input=train_file, epoch=20, lr=1, wordNgrams=1, verbose=2, minCount=10)
    ftu.save_model(model_file, model)


def test(test_file, model_file):
    textarr, labelarr = list(), list()
    with open(test_file) as testfp:
        lines = testfp.readlines()[:20]
    for line in lines:
        label, text = line.strip().split(' ', 1)
        textarr.append(text)
        labelarr.append(label)
    # for idx, line in enumerate(testlines):
    #     if pu.is_empty_string(line):
    #         continue
    #     label, text = line.split(' ', 1)
    #     print(label, model.predict(text, threshold=0.5), text)
    pred_value_arr = predict(textarr, ftu.load_model(model_file))
    label = [label2value[label] for label in labelarr]
    print(au.score(label, pred_value_arr, 'auc'))


if __name__ == '__main__':
    train_file = '/home/nfs/cdong/tw/seeding/NorthKorea/data/train'
    test_file = '/home/nfs/cdong/tw/seeding/NorthKorea/data/test'
    
    # train(train_file, korea_model_file)
    test(test_file, korea_model_file)
