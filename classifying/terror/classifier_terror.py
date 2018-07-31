import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import GradientBoostingClassifier

from config.configure import getcfg
import classifying.fast_text_utils as ftu
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.pattern_utils as pu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.timer_utils as tmu


value_t, value_f = ftu.value_t, ftu.value_f
terror_ft_model_file = getcfg().terror_ft_model_file
terror_clf_model_file = getcfg().terror_lr_model_file


class ClassifierTerror:
    nlp = None
    sensitive_words = {'shooting', 'wounded', 'shots', 'attack', 'shooter', 'wounds', 'dead',
                       'terrorist', 'hurt', 'terror', 'police', 'killed', 'gunman', 'weapon',
                       'injured', 'attacked', 'bomb', 'bombed', 'attacker'}
    
    # @staticmethod
    # def get_nlp():
    #     if ClassifierTerror.nlp is None:
    #         ClassifierTerror.nlp = su.get_nlp_disable_for_ner()
    
    def __init__(self, ft_model_file=terror_ft_model_file, clf_model_file=terror_clf_model_file):
        self.ft_model = self.clf_model = None
        if ft_model_file:
            self.load_ft_model(ft_model_file)
        if clf_model_file:
            self.load_clf_model(clf_model_file)
    
    def save_ft_model(self, ft_model_file):
        ftu.save_model(ft_model_file, self.ft_model)
    
    def load_ft_model(self, ft_model_file):
        self.ft_model = ftu.load_model(ft_model_file)
    
    def save_clf_model(self, clf_model_file):
        joblib.dump(self.clf_model, clf_model_file)
    
    def load_clf_model(self, clf_model_file):
        self.clf_model = joblib.load(clf_model_file)
    
    def get_ft_vector(self, text):
        """
        调用fasttext模型接口，返回文本的向量化表示
        :param text: str，普通文本
        :return: np.array，见 classifying.terror.classifier_terror.get_sentence_vector
        """
        return self.ft_model.get_sentence_vector(text)
    
    # def has_locate_feature(self, doc):
    #     for ent in doc.ents:
    #         if ent.label_ == 'GPE':
    #             return 1
    #     return -1
    
    def has_keyword_feature(self, text):
        """
        给定文本，按空格分词后，判断各分词是否是本类型事件的敏感词/触发词，进行计数
        :param text: str，普通文本
        :return: int，计数值
        """
        count = 0
        for word in text.lower().split():
            if word.strip() in ClassifierTerror.sensitive_words:
                count += 1
        return count
    
    def train_ft_model(self, train_file, **kwargs):
        """
        给定训练文件以及各项参数，调用fasttext的接口进行模型训练
        :param train_file: str，训练用文本文件的路径
        :param kwargs: fasttext所需关键字参数
        :return:
        """
        self.ft_model = ftu.FastText()
        self.ft_model.train_supervised(train_file, **kwargs)
    
    def train_clf_model(self, x, y, **kwargs):
        """
        给定向量组x以及各向量对应的标签y，调用模型接口进行训练
        :param x: np.array（2-d），每个元素为np.array（1-d），见 self.textarr2featurearr_no_gpe
        :param y: np.array（1-d），每个元素为int，表示x中对应位置的向量的标记
        :param kwargs: dict，分类器模型训练所需关键字参数
        :return:
        """
        self.clf_model = LogisticRegressionCV(**kwargs)
        self.clf_model.fit(x, y)
    
    def textarr2featurearr_no_gpe(self, textarr):
        """
        将输入文本列表转化为向量列表；其中，每条文本的向量除了fasttext提供的向量之外，
        还拼接了 self.has_keyword_feature 返回的敏感词/触发词计数值
        :param textarr: list，每个元素为str，普通文本
        :return: np.array（2-d），每个元素为np.array（1-d）
        """
        vecarr = list()
        for text in textarr:
            try:
                ft_vec = self.get_ft_vector(text)
            except:
                text = pu.text_normalization(text)
                ft_vec = self.get_ft_vector(text)
            ft_vec = np.append(ft_vec, self.has_keyword_feature(text))
            vecarr.append(ft_vec)
        return np.array(vecarr)
    
    # def textarr2featurearr(self, textarr, docarr):
    #     assert len(textarr) == len(docarr)
    #     vecarr = list()
    #     for idx in range(len(textarr)):
    #         text, doc = textarr[idx], docarr[idx]
    #         try:
    #             ft_vec = self.get_ft_vector(text)
    #         except:
    #             text = pu.text_normalization(text)
    #             ft_vec = self.get_ft_vector(text)
    #         ft_vec = np.append(ft_vec, self.has_locate_feature(doc))
    #         ft_vec = np.append(ft_vec, self.has_keyword_feature(text))
    #         vecarr.append(ft_vec)
    #     assert len(docarr) == len(vecarr)
    #     return np.array(vecarr)
    
    def predict_proba(self, featurearr):
        """
        给定一个向量列表，调用 self.clf_model 给出每个向量预测为正的概率值
        :param featurearr: np.array（2-d），每个元素为np.array（1-d），见self.textarr2featurearr_no_gpe
        :return: list，每个元素为float，表示 featurearr 中对应位置上的向量预测为正例的概率值
        """
        return list(self.clf_model.predict_proba(featurearr)[:, 1])
    
    def predict_mean_proba(self, twarr):
        """
        给定一个推特列表，调用 self.textarr2featurearr_no_gpe 得到向量列表后，
        计算均值向量，并调用 self.predict_proba 预测该均值向量为正例的概率值
        :param twarr: list，推特列表
        :return: float，预测概率值
        """
        textarr = [tw.get(tk.key_text) for tw in twarr]
        featurearr = self.textarr2featurearr_no_gpe(textarr)
        mean_feature = np.mean(featurearr, axis=0).reshape([1, -1])
        return self.clf_model.predict_proba(mean_feature)[0, 1]
    
    def filter(self, twarr, threshold):
        """
        给定推特列表，预测每条推特为正例的概率值，仅保留其中概率高于阈值的推特
        :param twarr: list，推特列表
        :param threshold: float，阈值
        :return: list，过滤后的推特列表
        """
        textarr = [tw.get(tk.key_text) for tw in twarr]
        """"""
        # docarr = su.textarr_nlp(textarr, self.get_nlp())
        # featurearr = self.textarr2featurearr(textarr, docarr)
        featurearr = self.textarr2featurearr_no_gpe(textarr)
        """"""
        probarr = self.predict_proba(featurearr)
        assert len(twarr) == len(probarr)
        return [tw for idx, tw in enumerate(twarr) if probarr[idx] >= threshold]
    
    def train_ft(self, train_file, ft_args, ft_model_file):
        """
        训练fasttext模型，并输出模型到文件
        :param train_file: str，训练用文本文件的路径
        :param ft_args: dict，fasttext 训练所需关键字参数
        :param ft_model_file: str，用于保存训练完成的模型的文件路径
        :return:
        """
        self.train_ft_model(train_file, **ft_args)
        self.save_ft_model(ft_model_file)
    
    def train_clf(self, featurearr, labelarr, clf_args, clf_model_file):
        """
        训练分类器，并输出模型到文件
        :param featurearr: np.array（2-d），每个元素为np.array（1-d），见 self.textarr2featurearr_no_gpe
        :param labelarr: np.array（1-d），每个元素为int，表示 featurearr 中对应位置的向量的标记
        :param clf_args: dict，分类器所需关键词参数
        :param clf_model_file: str，用于保存训练完成的模型的文件路径
        :return:
        """
        print(featurearr.shape, labelarr.shape, np.mean(labelarr))
        self.train_clf_model(featurearr, labelarr, **clf_args)
        self.save_clf_model(clf_model_file)
    
    def test(self, test_file):
        """
        给定带标记和文本的文件，读取其中的文本-标记对，调用向量化接口以及分类器，评估分类器在测试集上的性能
        :param test_file: str，测试用文本文件的路径
        :return:
        """
        textarr, labelarr = file2label_text_array(test_file)
        """"""
        # docarr = su.textarr_nlp(textarr, self.get_nlp())
        # featurearr = self.textarr2featurearr(textarr, docarr)
        featurearr = self.textarr2featurearr_no_gpe(textarr)
        """"""
        probarr = self.predict_proba(featurearr)
        au.precision_recall_threshold(labelarr, probarr,
            thres_range=[i / 100 for i in range(1, 10)] + [i / 20 for i in range(2, 20)])
        fu.dump_array("result.json", (labelarr, probarr))


def file2label_text_array(file):
    """
    转化文本-标记文件到内存
    :param file: str，文本-标记文件的路径
    :return: 见 text2label_text_array
    """
    lines = fu.read_lines(file)
    return text2label_text_array(lines)


def text2label_text_array(lbl_txt_lines):
    """
    将文本-标记文本列表的每条文本按空格拆分为文本-标记tuple
    :param lbl_txt_lines: list，每个元素为str，文本的第一个空格之前为标记，之后为文本内容
    :return:
        textarr: list，每个元素为str，即文本内容
        labelarr: list，每个元素为int，即文本对应的标记，0表示为事件负例，1表示为事件正例
    """
    label2value = ftu.binary_label2value
    textarr, labelarr = list(), list()
    for line in lbl_txt_lines:
        try:
            label, text = line.strip().split(' ', maxsplit=1)
            label = label2value[label]
        except:
            raise ValueError('Label wrong, check the text file.')
        textarr.append(text)
        labelarr.append(label)
    print('total text length: {}'.format(len(textarr)))
    return textarr, labelarr


def generate_train_matrices(ft_model_file, lbl_txt_file, mtx_lbl_file_list):
    """
    给出fasttext模型文件的路径，读取文本-标记文件，将文件内容分块传递给多个子进程，
    各子进程将文本和标记分别转化为向量列表（即矩阵）输出到 mtx_lbl_file_list 中的每个文件中
    文本量较大的情况下避免每次训练分类器都要重新生成文本对应的向量列表
    :param ft_model_file: str，fasttext模型的文件路径
    :param lbl_txt_file: str，文本-标记文件的路径
    :param mtx_lbl_file_list: 每个元素为tuple，tuple的每个元素为str，
        第一个str标志存储矩阵的文件，第二个str表示存储该矩阵对应的标记列表的文件
    :return:
    """
    lbl_txt_arr = fu.read_lines(lbl_txt_file)
    lbl_txt_blocks = mu.split_multi_format(lbl_txt_arr, len(mtx_lbl_file_list))
    args_list = [(ft_model_file, lbl_txt_blocks[idx], mtx_file, lbl_file)
                 for idx, (mtx_file, lbl_file) in enumerate(mtx_lbl_file_list)]
    print([len(b) for b in lbl_txt_blocks])
    mu.multi_process_batch(_generate_matrices, 10, args_list)


def _generate_matrices(ft_model_file, lbl_txt_arr, mtx_file, lbl_file):
    """
    读取fasttext模型，给出文本-标记对列表，将文本转化为矩阵存储到 mtx_file 中，将标记转化为矩阵存储到 lbl_file 中
    :param ft_model_file: str，fasttext模型的文件路径
    :param lbl_txt_arr: list，每个元素为str，交由 text2label_text_array 处理
    :param mtx_file: str，矩阵的输出路径
    :param lbl_file: str，标记的输出路径
    :return:
    """
    print(len(lbl_txt_arr), mtx_file, lbl_file)
    textarr, labelarr = text2label_text_array(lbl_txt_arr)
    clf = ClassifierTerror(ft_model_file, None)
    """"""
    # docarr = su.textarr_nlp(textarr, clf.get_nlp())
    # tmu.check_time('_generate_matrices')
    # featurearr = clf.textarr2featurearr(textarr, docarr)
    featurearr = clf.textarr2featurearr_no_gpe(textarr)
    """"""
    np.save(mtx_file, featurearr)
    np.save(lbl_file, labelarr)


def recover_train_matrix(mtx_lbl_file_list):
    """
    从文件列表中读取矩阵记忆标记到内存
    :param mtx_lbl_file_list: 每个元素为tuple，tuple的每个元素为str，
        第一个str标志存储矩阵的文件，第二个str表示存储该矩阵对应的标记列表的文件
    :return:
        featurearr: 按顺序读取各矩阵文件内容后拼接形成的大矩阵
        labelarr: 按顺序读取各标记文件内容后拼接形成的大矩阵
    """
    mtx_list, lbl_list = list(), list()
    for idx, (mtx_file, lbl_file) in enumerate(mtx_lbl_file_list):
        print("recovering {} / {}".format(idx + 1, len(mtx_lbl_file_list)))
        mtx_list.append(np.load(mtx_file))
        lbl_list.append(np.load(lbl_file))
    featurearr = np.concatenate(mtx_list, axis=0)
    labelarr = np.concatenate(lbl_list, axis=0)
    return featurearr, labelarr


def coef_of_lr_model(file):
    from sklearn.externals import joblib
    lr_model = joblib.load(file)
    print(len(lr_model.coef_[0]))
    print(lr_model.coef_)


if __name__ == "__main__":
    from classifying.terror.data_maker import fasttext_train, fasttext_test, ft_data_pattern
    ft_model = "/home/nfs/cdong/tw/src/models/classify/terror/ft_no_gpe_model"
    lr_model = "/home/nfs/cdong/tw/src/models/classify/terror/lr_no_gpe_model"
    clf_model = lr_model
    tmu.check_time('all')
    tmu.check_time()
    
    # coef_of_lr_model("/home/nfs/cdong/tw/src/models/classify/terror/lr_no_gpe_model")
    # clf_filter = ClassifierAddFeature(None, None)
    # for file in fi.listchildren("/home/nfs/cdong/tw/seeding/Terrorist/queried/positive", concat=True):
    #     twarr = fu.load_array(file)
    #     print(file, clf_filter.predict_mean_proba(twarr))
    # tmu.check_time()
    # exit()
    
    batch_num = 20
    fi.mkdir(ft_data_pattern.format('matrices_no_add'), remove_previous=True)
    train_mtx_ptn = ft_data_pattern.format('matrices_no_add/train_feature_mtx_{}.npy')
    train_lbl_ptn = ft_data_pattern.format('matrices_no_add/train_lblarr_mtx_{}.npy')
    train_file_list = [(train_mtx_ptn.format(idx), train_lbl_ptn.format(idx)) for idx in range(batch_num)]
    
    # _clf = ClassifierAddFeature(None, None)
    # _ft_args = dict(epoch=150, lr=1.5, wordNgrams=2, verbose=2, minCount=2, thread=20, dim=300)
    # tmu.check_time()
    
    # _clf.train_ft(fasttext_train, _ft_args, ft_model)
    # tmu.check_time(print_func=lambda dt: print('train ft time: {}s'.format(dt)))
    # generate_train_matrices(ft_model, fasttext_train, train_file_list)
    # tmu.check_time()
    # _featurearr, _labelarr = recover_train_matrix(train_file_list)
    # tmu.check_time()
    #
    # _lr_args = dict(n_jobs=20, max_iter=300, tol=1e-6, class_weight={value_f: 1, value_t: 10})
    # _clf.train_clf(_featurearr, _labelarr, _lr_args, clf_model)
    # tmu.check_time(print_func=lambda dt: print('train lr time: {}s'.format(dt)))
    
    _clf = ClassifierTerror(ft_model, clf_model)
    _clf.test(fasttext_test)
    tmu.check_time(print_func=lambda dt: print('test time: {}s'.format(dt)))
    tmu.check_time('all')
