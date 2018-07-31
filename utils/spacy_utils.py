import spacy
import numpy as np


"""
[(token.text, token.ent_type_, token.tag_, token.pos_, ) for token in doc]
[(ent.text, ent.root.tag_, ent.label_) for ent in doc.ents]
"""


default_model_name = 'en_core_web_lg'
nlp_dict = {}
glovec_dim = 300


def get_nlp_disable_for_ner(model_key=default_model_name):
    # this does not harm the performance of NER, but can reduce time by about 20%
    return spacy.load(model_key, disable=['tagger'])


def new_nlp(model):
    print("reading model {}".format(model))
    return spacy.load(model)


def get_nlp(model_key=default_model_name):
    if model_key not in nlp_dict:
        nlp_dict[model_key] = new_nlp(model_key)
    return nlp_dict[model_key]


def text_nlp(text, nlp=None):
    nlp = get_nlp() if nlp is None else nlp
    return nlp(text)


def textarr_nlp(textarr, nlp=None, n_threads=4):
    nlp = get_nlp() if nlp is None else nlp
    return [doc for doc in nlp.pipe(textarr, n_threads=n_threads)]


pos_prop = 'PROPN'
pos_comm = 'NOUN'
pos_verb = 'VERB'
pos_hstg = 'HSTG'

LABEL_GPE = 'GPE'
LABEL_FAC = 'FAC'
LABEL_LOC = 'LOC'
LABEL_LOCATION = {LABEL_GPE, LABEL_FAC, LABEL_LOC}
target_pos_types = [pos_hstg, pos_prop, pos_comm, pos_hstg]
target_pos_sets = set(target_pos_types)


# def get_doc_pos_vectors(doc):
#     """ returns embedding vector featuring the doc using above types of pos tags """
#     postype2itemarr = dict([(pos_type, list()) for pos_type in target_pos_types])
#     for token in doc:
#         if not (token.has_vector and token.pos_ in postype2itemarr):
#             continue
#         postype2itemarr[token.pos_].append([token.tag_, token.vector])
#     vecarr = list()
#     for pos_type in target_pos_types:
#         itemarr = postype2itemarr[pos_type]
#         if len(itemarr) == 0:
#             pos_vector = np.zeros([glovec_dim, ])
#         else:
#             pos_vector = np.mean([item[1] for item in itemarr], axis=0)
#         assert len(pos_vector) == glovec_dim
#         vecarr.append(pos_vector)
#     return vecarr


if __name__ == '__main__':
    import utils.function_utils as fu
    import utils.timer_utils as tmu
    
    pos_file = "/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/pos_2016.txt"
    txtarr = fu.read_lines(pos_file)
    
    nlp1 = spacy.load("en_core_web_lg")
    nlp2 = spacy.load("en_core_web_lg", disable=['tagger'])
    nlp3 = spacy.load("en_core_web_lg", disable=['parser'])
    nlp4 = spacy.load("en_core_web_lg", disable=['parser', 'tagger'])
    
    tmu.check_time()
    docarr1 = list(nlp1.pipe(txtarr, n_threads=10))
    tmu.check_time()
    docarr2 = list(nlp2.pipe(txtarr, n_threads=10))
    tmu.check_time()
    docarr3 = list(nlp3.pipe(txtarr, n_threads=10))
    tmu.check_time()
    docarr4 = list(nlp4.pipe(txtarr, n_threads=10))
    tmu.check_time()
    
    assert len(docarr1) == len(docarr2)
    diff_cnt = 0
    for idx in range(len(docarr1)):
        txt, doc1, doc2, doc3, doc4 = txtarr[idx], docarr1[idx], docarr2[idx], docarr3[idx], docarr4[idx]
        # txt, doc1, doc2 = txtarr[idx], docarr1[idx], docarr2[idx]
        # print(txt)
        ent1 = ','.join([ent.text for ent in doc1.ents if ent.label_ in LABEL_LOCATION])
        ent2 = ','.join([ent.text for ent in doc2.ents if ent.label_ in LABEL_LOCATION])
        ent3 = ','.join([ent.text for ent in doc3.ents if ent.label_ in LABEL_LOCATION])
        ent4 = ','.join([ent.text for ent in doc4.ents if ent.label_ in LABEL_LOCATION])
        # print(ent1)
        # print(ent2)
        # print(ent3)
        # print('--\n')
        if not ent1 == ent2 == ent3 == ent4:
            print(txt)
            print(1, ent1)
            print(2, ent2)
            print(3, ent3)
            print(4, ent4)
            print('--')
            diff_cnt += 1
    print("diffffff", diff_cnt)
