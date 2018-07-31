import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import utils.function_utils as fu


def figure(X, Y, fig_name):
    # plt.figure(figsize=(13, 7))
    plt.plot(X, Y, color="blue", linewidth=1)
    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("roc curve")
    plt.legend(loc='lower right')
    plt.savefig(fig_name, format='png')


if __name__ == '__main__':
    import re
    import numpy as np
    from pathlib import Path
    import utils.file_iterator as fi
    files = fi.listchildren('/home/nfs/cdong/tw/testdata/output2', children_type=fi.TYPE_DIR, concat=True)
    ls = list()
    for f in files:
        p = Path(f)
        s = p.stat()
        digits = re.findall('\d+', p.name)
        cluid, clunum = list(map(int, digits))
        ls.append((cluid, clunum, s.st_mtime))
    
    ls = sorted(ls, key=lambda item: item[0])
    print(ls)
    dt = [(ls[0][0], ls[0][1], 0)] + [(ls[i][0], ls[i][1], int(ls[i][2] - ls[i - 1][2])) for i in range(1, len(ls))]
    print(dt)

    # x, y_cnum, y_dt = list(zip(*dt[:50]))
    x, y_cnum, y_dt = list(zip(*dt))
    y_cnum_norm = np.array(y_cnum) / np.max(y_cnum)
    y_dt_norm = np.array(y_dt) / np.max(y_dt)
    print('time sum: {}'.format(np.sum(y_dt)))
    print('max cnum:{}, min cnum:{}'.format(max(y_cnum), min(y_cnum)))
    print('max dt:{}, min dt:{}'.format(max(y_dt), min(y_dt)))
    
    plt.figure(figsize=(20, 10))
    plt.plot(x, y_cnum_norm, ".-", color="blue", linewidth=0.5)
    plt.plot(x, y_dt_norm, "x-", color="red", linewidth=0.5)
    plt.savefig('test_zhexian.png')
    plt.close()

    # import utils.array_utils as au
    # labelarr, probarr = fu.load_array("/home/nfs/cdong/tw/src/preprocess/filter/prb_lbl_arr.txt")
    # from sklearn.metrics import roc_curve
    # fpr, tpr, thresholds = roc_curve(labelarr, probarr)
    # figure(fpr, tpr, "/home/nfs/cdong/tw/src/preprocess/filter/roc_curve.png")
    # au.precision_recall_threshold(labelarr, probarr, file="/home/nfs/cdong/tw/src/preprocess/filter/performance.csv",
    #                               thres_range=[i / 100 for i in range(1, 10)] + [i / 20 for i in range(2, 20)])
