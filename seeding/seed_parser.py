import utils.file_iterator as fi
import utils.function_utils as fu
from seeding.seed_query import SeedQuery


class SeedParser:
    def __init__(self, query_list, theme, description):
        self.base_path = './'
        self.theme = theme
        self.description = description
        
        self.query_list = query_list
        self.seed_query_list = [SeedQuery(*query) for query in query_list]
        
        self.added_ids = set()
        self.added_twarr = list()
    
    def read_tweet_from_json_file(self, file):
        if not self.is_file_of_query_date(file):
            return
        for tw in fu.load_array(file):
            tw_added = False
            for seed_query in self.seed_query_list:
                tw_added = seed_query.append_desired_tweet(tw, usingtwtime=False) or tw_added
            if tw_added:
                if tw['id'] in self.added_ids:
                    continue
                else:
                    self.added_ids.add(tw['id'])
                self.added_twarr.append(tw)
    
    def is_file_of_query_date(self, file_name):
        for seed_query in self.seed_query_list:
            tw_ymd = seed_query.time_of_tweet(file_name, source='filename')
            if seed_query.is_time_desired(tw_ymd):
                return True
        return False
    
    def get_query_results(self): return self.added_twarr
    
    def set_base_path(self, base_path):
        self.base_path = fi.add_sep_if_needed(base_path)
        for path in [self.get_base_path(), self.get_theme_path(), self.get_queried_path(), self.get_param_path(), ]:
            fi.mkdir(path)
    
    def get_base_path(self): return self.base_path
    
    def get_theme_path(self): return fi.join(self.get_base_path(), self.theme)
    
    def get_queried_path(self): return fi.join(self.get_theme_path(), 'queried')
    
    def get_param_path(self): return fi.join(self.get_theme_path(), 'params')
    
    def get_query_result_file_name(self): return self.get_queried_path() + self.theme + '.sum'
    
    # def get_param_file_name(self): return fi.join(self.get_param_path(), self.theme)
    
    # def get_dict_file_name(self): return fi.join(self.get_param_path(), self.theme + '.dic')


# class UnlbParser(SeedParser):
#     def __init__(self, query_list, theme, description):
#         SeedParser.__init__(self, query_list, theme, description)
#
#     def get_query_result_file_name(self):
#         return self.get_queried_path() + self.theme + '_unlabelled.sum'
#
#
# class CounterParser(SeedParser):
#     def __init__(self, query_list, theme, description):
#         SeedParser.__init__(self, query_list, theme, description)
#
#     def get_query_result_file_name(self):
#         return self.get_queried_path() + self.theme + '_counter.sum'
#
#
# class TestParser(SeedParser):
#     def __init__(self, query_list, theme, description, outterid='default'):
#         SeedParser.__init__(self, query_list, theme, description)
#         self.outterid = outterid
#
#     def get_query_result_file_name(self):
#         return self.get_queried_path() + self.theme + '_test_' + self.outterid + '.sum'
