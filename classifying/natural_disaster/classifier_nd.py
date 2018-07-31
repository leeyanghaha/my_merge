import numpy as np
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import classifying.fast_text_utils as ftu
import utils.multiprocess_utils as mu
import utils.pattern_utils as pu
import utils.tweet_keys as tk
import utils.timer_utils as tmu
from config.configure import getcfg

from classifying.terror.classifier_terror import \
    ClassifierTerror, file2label_text_array, text2label_text_array


value_t, value_f = ftu.value_t, ftu.value_f
nd_ft_model_file = getcfg().nd_ft_model_file
nd_clf_model_file = getcfg().nd_lr_model_file


class ClassifierNaturalDisaster(ClassifierTerror):
    def __init__(self, ft_model_file=nd_ft_model_file, clf_model_file=nd_clf_model_file):
        ClassifierTerror.__init__(self, ft_model_file, clf_model_file)
        self.ft_model = self.clf_model = None
        if ft_model_file:
            self.load_ft_model(ft_model_file)
        if clf_model_file:
            self.load_clf_model(clf_model_file)
    
    def textarr2featurearr(self, textarr):
        vecarr = list()
        for text in textarr:
            try:
                ft_vec = self.get_ft_vector(text)
            except:
                text = pu.text_normalization(text)
                ft_vec = self.get_ft_vector(text)
            vecarr.append(ft_vec)
        return np.array(vecarr)
    
    def predict_mean_proba(self, twarr):
        textarr = [tw.get(tk.key_text) for tw in twarr]
        featurearr = self.textarr2featurearr(textarr)
        mean_feature = np.mean(featurearr, axis=0).reshape([1, -1])
        return self.clf_model.predict_proba(mean_feature)[0, 1]
    
    def filter(self, twarr, threshold):
        textarr = [tw.get(tk.key_text) for tw in twarr]
        featurearr = self.textarr2featurearr(textarr)
        probarr = self.predict_proba(featurearr)
        assert len(twarr) == len(probarr)
        return [tw for idx, tw in enumerate(twarr) if probarr[idx] >= threshold]
    
    def test(self, test_file):
        textarr, labelarr = file2label_text_array(test_file)
        featurearr = self.textarr2featurearr(textarr)
        probarr = self.predict_proba(featurearr)
        au.precision_recall_threshold(labelarr, probarr, file="performance.csv",
                                      thres_range=[i / 100 for i in range(1, 10)] + [i / 20 for i in range(2, 20)])
        fu.dump_array("result.json", (labelarr, probarr))


def generate_train_matrices(ft_model_file, lbl_txt_file, mtx_lbl_file_list):
    lbl_txt_arr = fu.read_lines(lbl_txt_file)
    lbl_txt_blocks = mu.split_multi_format(lbl_txt_arr, len(mtx_lbl_file_list))
    args_list = [(ft_model_file, lbl_txt_blocks[idx], mtx_file, lbl_file)
                 for idx, (mtx_file, lbl_file) in enumerate(mtx_lbl_file_list)]
    print([len(b) for b in lbl_txt_blocks])
    mu.multi_process_batch(_generate_matrices, 20, args_list)


def _generate_matrices(ft_model_file, lbl_txt_arr, mtx_file, lbl_file):
    print(len(lbl_txt_arr), mtx_file, lbl_file)
    textarr, labelarr = text2label_text_array(lbl_txt_arr)
    clf = ClassifierNaturalDisaster(ft_model_file, None)
    featurearr = clf.textarr2featurearr(textarr)
    np.save(mtx_file, featurearr)
    np.save(lbl_file, labelarr)


def recover_train_matrix(mtx_lbl_file_list):
    mtx_list, lbl_list = list(), list()
    for idx, (mtx_file, lbl_file) in enumerate(mtx_lbl_file_list):
        print("recovering {} / {}".format(idx + 1, len(mtx_lbl_file_list)))
        mtx_list.append(np.load(mtx_file))
        lbl_list.append(np.load(lbl_file))
    featurearr = np.concatenate(mtx_list, axis=0)
    labelarr = np.concatenate(lbl_list, axis=0)
    return featurearr, labelarr


if __name__ == "__main__":
    from classifying.natural_disaster.nd_data_maker import nd_train, nd_test, nd_data_pattern
    nd_ft_model = "/home/nfs/cdong/tw/src/models/classify/natural_disaster/ft_model"
    nd_lr_model = "/home/nfs/cdong/tw/src/models/classify/natural_disaster/lr_model"
    tmu.check_time('all')
    tmu.check_time()
    
    batch_num = 20
    fi.mkdir(nd_data_pattern.format('matrices'))
    nd_train_mtx = nd_data_pattern.format('matrices/train_feature_mtx_{}.npy')
    nd_train_lbl = nd_data_pattern.format('matrices/train_lblarr_mtx_{}.npy')
    nd_train_mtx_files = [(nd_train_mtx.format(idx), nd_train_lbl.format(idx)) for idx in range(batch_num)]
    
    _clf = ClassifierNaturalDisaster(None, None)
    # _ft_args = dict(epoch=150, lr=1.5, wordNgrams=2, verbose=2, minCount=2, thread=20, dim=250)
    # _clf.train_ft(nd_train, _ft_args, nd_ft_model)
    # tmu.check_time(print_func=lambda dt: print('train ft time: {}s'.format(dt)))
    generate_train_matrices(nd_ft_model, nd_train, nd_train_mtx_files)
    tmu.check_time()
    _featurearr, _labelarr = recover_train_matrix(nd_train_mtx_files)
    tmu.check_time()
    
    _lr_args = dict(n_jobs=20, max_iter=300, tol=1e-6, class_weight={value_f: 1, value_t: 10})
    _clf.train_clf(_featurearr, _labelarr, _lr_args, nd_lr_model)
    tmu.check_time(print_func=lambda dt: print('train lr time: {}s'.format(dt)))
    
    _clf = ClassifierNaturalDisaster(nd_ft_model, nd_lr_model)
    _clf.test(nd_test)
    tmu.check_time(print_func=lambda dt: print('test time: {}s'.format(dt)))
    tmu.check_time('all')
