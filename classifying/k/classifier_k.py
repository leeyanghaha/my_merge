import numpy as np
import utils.file_iterator as fi
import classifying.fast_text_utils as ftu
import utils.timer_utils as tmu
from config.configure import getcfg

from classifying.natural_disaster.classifier_nd import \
    ClassifierNaturalDisaster, text2label_text_array, recover_train_matrix, generate_train_matrices


value_t, value_f = ftu.value_t, ftu.value_f
k_ft_model_file = getcfg().k_ft_model_file
k_clf_model_file = getcfg().k_lr_model_file


class ClassifierK(ClassifierNaturalDisaster):
    def __init__(self, ft_model_file=k_ft_model_file, clf_model_file=k_clf_model_file):
        ClassifierNaturalDisaster.__init__(self, ft_model_file, clf_model_file)


def _generate_matrices(ft_model_file, lbl_txt_arr, mtx_file, lbl_file):
    print(len(lbl_txt_arr), mtx_file, lbl_file)
    textarr, labelarr = text2label_text_array(lbl_txt_arr)
    clf = ClassifierK(ft_model_file, None)
    featurearr = clf.textarr2featurearr(textarr)
    np.save(mtx_file, featurearr)
    np.save(lbl_file, labelarr)


if __name__ == "__main__":
    from classifying.k.k_data_maker import train_file, test_file, k_data_pattern
    ft_model = "/home/nfs/cdong/tw/src/models/classify/korea/ft_model"
    lr_model = "/home/nfs/cdong/tw/src/models/classify/korea/lr_model"
    tmu.check_time('all')
    tmu.check_time()
    
    batch_num = 20
    fi.mkdir(k_data_pattern.format('matrices'))
    train_mtx = k_data_pattern.format('matrices/train_feature_mtx_{}.npy')
    train_lbl = k_data_pattern.format('matrices/train_lblarr_mtx_{}.npy')
    train_mtx_files = [(train_mtx.format(idx), train_lbl.format(idx)) for idx in range(batch_num)]
    
    _clf = ClassifierK(None, None)
    _ft_args = dict(epoch=150, lr=1.5, wordNgrams=2, verbose=2, minCount=2, thread=20, dim=250)
    _clf.train_ft(train_file, _ft_args, ft_model)
    tmu.check_time(print_func=lambda dt: print('train ft time: {}s'.format(dt)))
    generate_train_matrices(ft_model, train_file, train_mtx_files)
    tmu.check_time()
    _featurearr, _labelarr = recover_train_matrix(train_mtx_files)
    tmu.check_time()
    
    _lr_args = dict(n_jobs=20, max_iter=300, tol=1e-6, class_weight={value_f: 1, value_t: 10})
    _clf.train_clf(_featurearr, _labelarr, _lr_args, lr_model)
    tmu.check_time(print_func=lambda dt: print('train lr time: {}s'.format(dt)))
    
    _clf = ClassifierK(ft_model, lr_model)
    _clf.test(test_file)
    tmu.check_time(print_func=lambda dt: print('test time: {}s'.format(dt)))
    tmu.check_time('all')
