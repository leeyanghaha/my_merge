import argparse
from config.configure import getcfg
import utils.function_utils as fu
from preprocess import summarization


@fu.sync_real_time_counter('preprocess')
def main():
    # fi.iterate_file_tree(getconfig().data_path, summary_files_in_path,
    #                      summary_path='/home/nfs/cdong/tw/testdata/cdong/non')
    
    # summarization.get_semantic_tokens_multi(getcfg().origin_path)
    
    summarization.get_tokens_multi(getcfg().origin_path)
    
    # summarization.summary_files_in_path_into_blocks(
    #     '/home/nfs/cdong/tw/testdata/cdong/events/event2012origin/cut_by_id', './', 'events2012.txt')


def parse_args():
    parser = argparse.ArgumentParser(description="preprocess")
    # parser.add_argument('--simp', action='store_true', default=False, help='simplify_files_multi')
    return parser.parse_args()


if __name__ == '__main__':
    main()
