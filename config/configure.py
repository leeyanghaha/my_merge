import os
from configparser import ConfigParser, ExtendedInterpolation


class Configure:
    def __init__(self):
        path_module = 'path'
        file_module = 'files'
        mongodb = 'mongodb'
        file_dir = os.path.abspath(os.path.dirname(__file__))
        # file_dir = os.path.split(os.path.realpath(__file__))[0]
        config_file_name = os.path.join(file_dir, 'conf.ini')
        configreader = ConfigParser(interpolation=ExtendedInterpolation())
        configreader.read(config_file_name)
        
        # path
        self.autophrase_path = configreader.get(path_module, 'autophrase_path')
        self.sutime_jar_path = configreader.get(path_module, 'sutime_jar_path')
        self.output_path = configreader.get(path_module, 'output_path')
        
        # files
        self.post_dict_file = configreader.get(file_module, 'post_dict_file')
        
        self.afinn_file = configreader.get(file_module, 'afinn_file')
        self.black_list_file = configreader.get(file_module, 'black_list_file')
        
        self.clf_model_file = configreader.get(file_module, 'clf_model_file')
        self.class_dist_file = configreader.get(file_module, 'class_dist_file')
        self.chat_filter_file = configreader.get(file_module, 'chat_filter_file')
        self.is_noise_dict_file = configreader.get(file_module, 'is_noise_dict_file')
        self.orgn_predict_label_file = configreader.get(file_module, 'orgn_predict_label_file')
        
        self.terror_ft_model_file = configreader.get(file_module, 'terror_ft_model_file')
        self.terror_lr_model_file = configreader.get(file_module, 'terror_lr_model_file')
        self.nd_ft_model_file = configreader.get(file_module, 'nd_ft_model_file')
        self.nd_lr_model_file = configreader.get(file_module, 'nd_lr_model_file')
        self.k_ft_model_file = configreader.get(file_module, 'k_ft_model_file')
        self.k_lr_model_file = configreader.get(file_module, 'k_lr_model_file')

        #db
        self.mongo2 = configreader.get(mongodb,'ubuntu2')

config = Configure()


def getcfg():
    return config


if __name__ == "__main__":
    print(getcfg().__dict__)
