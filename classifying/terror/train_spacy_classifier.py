from __future__ import unicode_literals, print_function

import math
from collections import Counter

import spacy
from spacy.util import minibatch, compounding
import utils.function_utils as fu
import utils.array_utils as au

import utils.spacy_utils as su


true_label = "is_terror"

pos_text_file = "/home/nfs/cdong/tw/seeding/Terrorist/data/spacy/pos.txt"   # 24001
neg_text_file = "/home/nfs/cdong/tw/seeding/Terrorist/data/spacy/neg.txt"   # 659435

trained_spacy_path = "/home/nfs/cdong/tw/src/models/spacy/"


def split_array(item_array, label):
    """ All items in item_array share the same label """
    split = int(len(item_array) * 0.9)
    label_array = [label] * len(item_array)
    train_x, test_x = item_array[:split], item_array[split:]
    train_y, test_y = label_array[:split], label_array[split:]
    print("len:{}, test len:{}".format(len(train_x), len(test_x)))
    return train_x, test_x, train_y, test_y


def evaluate(tokenizer, terror_cat, texts, cats):
    tokens_list = (tokenizer(text) for text in texts)
    labels, scores = list(), list()
    for i, doc in enumerate(terror_cat.pipe(tokens_list)):
        label = cats[i]['cats'][true_label]
        score = doc.cats[true_label]
        labels.append(label)
        scores.append(score)
    au.precision_recall_threshold(labels, scores)
    print('\n')


def train_spacy_model(nlp):
    p_textarr = fu.read_lines(pos_text_file)
    n_textarr = fu.read_lines(neg_text_file)[-10000:]
    p_train_x, p_test_x, p_train_y, p_test_y = split_array(p_textarr, {'cats': {true_label: True}})
    n_train_x, n_test_x, n_train_y, n_test_y = split_array(n_textarr, {'cats': {true_label: False}})
    
    p_train_data = list(zip(p_train_x, p_train_y))
    n_train_data = list(zip(n_train_x, n_train_y))
    train_data = p_train_data + n_train_data
    test_x, test_y = p_test_x + n_test_x, p_test_y + n_test_y
    
    """ prepare pipelines """
    vocab_size = len(nlp.vocab)
    pipe_cat_name = 'textcat'       # the pipe has to be named so. or spacy cannot recognize it
    if pipe_cat_name not in nlp.pipe_names:
        terror_cat = nlp.create_pipe(pipe_cat_name)
        nlp.add_pipe(terror_cat, last=True)
    else:
        terror_cat = nlp.get_pipe(pipe_cat_name)
    terror_cat.add_label(true_label)
    
    """ start training """
    n_iter = 10
    other_pipe_names = [pipe for pipe in nlp.pipe_names if pipe != pipe_cat_name]
    with nlp.disable_pipes(*other_pipe_names):   # only train textcat
        optimizer = nlp.begin_training()
        for i in range(n_iter):
            print("iter:{}".format(i))
            losses = {}
            batch_size = 16
            batch_num = int(math.ceil(len(train_data) / batch_size))
            batches = [train_data[idx * batch_size: (idx + 1) * batch_size] for idx in range(batch_num)]
            print(Counter([len(b) for b in batches]))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            print("losses:", losses)
            with terror_cat.model.use_params(optimizer.averages):
                evaluate(nlp.tokenizer, terror_cat, test_x, test_y)
    print("vocab size: {} -> {}".format(vocab_size, len(nlp.vocab)))
    return nlp


def dump_nlp(directory, nlp):
    from pathlib import Path
    output_dir = Path(directory)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)


def test_nlp(directory):
    text1 = "Another explosion has occurred not far from Paris."
    text2 = "It has beat me down, and I got deeply hurt."
    nlp = spacy.load(directory)
    doc1 = nlp(text1)
    print(text1, doc1.cats)
    doc2 = nlp(text2)
    print(text2, doc2.cats)


if __name__ == '__main__':
    import utils.timer_utils as tmu
    import utils.pattern_utils as pu
    import numpy as np

    # matrix_pre = np.load('matrix_pre.npy')
    # matrix_post = np.load('matrix_post.npy')
    # diff = matrix_pre - matrix_post
    # diff = np.square(diff)
    # print(np.sum(diff))
    # exit()
    
    # p_textarr = fu.read_lines(pos_text_file)
    # tokens_list = [pu.tokenize('[\_\w\-]{2,}', text) for text in p_textarr]
    # cnt = Counter(au.merge_array(tokens_list))
    # print(cnt.most_common(100))
    # exit()
    
    word_list = ['attack', 'bomb', 'bombing', 'kill', 'killed', 'explode', 'explosion', 'terrorist', 'suicide']
    
    _nlp = su.get_nlp()
    voacb = _nlp.vocab
    matrix_pre = np.array([voacb.get_vector(w) for w in word_list])
    np.save('matrix_pre', matrix_pre)
    
    tmu.check_time()
    _nlp = train_spacy_model(_nlp)
    tmu.check_time()
    # dump_nlp(trained_spacy_path, _nlp)
    # tmu.check_time()
    # test_nlp(trained_spacy_path)
    
    voacb = _nlp.vocab
    matrix_post = np.array([voacb.get_vector(w) for w in word_list])
    np.save('matrix_post', matrix_post)
    
    diff = matrix_pre - matrix_post
    diff = np.square(diff)
    print("diff:", np.sum(diff))


