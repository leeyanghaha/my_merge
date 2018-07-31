from utils.id_freq_dict import IdFreqDict
from config.configure import getcfg


class IfdGetter:
    K_IFD_FILE = 'ifd_file'
    
    def __init__(self, ifd_file=None):
        self.ifd_file = ifd_file
        self.ifd = None
    
    def __call__(self, *args, **kwargs):
        if IfdGetter.K_IFD_FILE in kwargs:
            self.ifd_file = kwargs.get(IfdGetter.K_IFD_FILE)
        if self.ifd_file is None:
            raise ValueError('An id freq dict should be specified.')
        if self.ifd is None:
            self.ifd = IdFreqDict()
            self.ifd.load_dict(self.ifd_file)
        return self.ifd
    
    def reload(self, ifd_file):
        if self.ifd is not None:
            self.ifd.load_dict(ifd_file)


# pre_dict_file = getcfg().pre_dict_file
post_dict_file = getcfg().post_dict_file
token_dict = IfdGetter(post_dict_file)

# pre_list = [getcfg().pre_prop_file, getcfg().pre_comm_file, getcfg().pre_verb_file, getcfg().pre_hstg_file]
# post_list = [getcfg().post_prop_file, getcfg().post_comm_file, getcfg().post_verb_file, getcfg().post_hstg_file]
# prop_dict, comm_dict, verb_dict, hstg_dict = [IfdGetter(post_file) for post_file in post_list]


if __name__ == '__main__':
    import utils.pattern_utils as pu
    
    def word_remove(word, freq):
        if pu.search_pattern(r'!?<>.,&\'`\^*', word) is not None or freq < 10:
            return True
        return False
    
    pre2post = dict(zip(pre_list, post_list))
    for pre, post in pre2post.items():
        ifd = IdFreqDict()
        ifd.load_dict(pre)
        pre_vocab = ifd.vocabulary_size()
        print('{} loaded, {} words'.format(pre, pre_vocab))
        ifd.drop_words_by_condition(word_remove)
        print('{} words dropped, remain {} words'.format(pre_vocab - ifd.vocabulary_size(), ifd.vocabulary_size()))
        ifd.dump_dict(post)
        print('dump over')
