import os
import shutil
import utils.pattern_utils as pu
from pathlib import Path


def add_sep_if_needed(path): return path if path.endswith(os.path.sep) else path + os.path.sep


def getcwd(): return os.getcwd()


def get_name(path): return Path(path).name


def base_name(abspath): return os.path.basename(abspath)


def get_parent_path(path): return os.path.dirname(path)


def is_dir(path): return os.path.isdir(path)


def is_file(file): return os.path.isfile(file)


def exists(path): return os.path.exists(path)


def rmtree(path):
    if os.path.exists(path):
        print('remove path {}'.format(path))
        shutil.rmtree(path)


def remove_file(file):
    if type(file) is str:
        if os.path.exists(file):
            os.remove(file)
    else:
        raise TypeError("File descriptor not an expected type")


def remove_files(files):
    if type(files) is list:
        for f in files:
            remove_file(f)
    else:
        raise TypeError("File descriptor array not an expected type")


def concat_files(file_list, output_file):
    input_files = ' '.join(file_list)
    p = os.popen('cat {} > {}'.format(input_files, output_file))
    p.close()


def mkdir(dir_name, remove_previous=False):
    if remove_previous:
        rmtree(dir_name)
    if not exists(dir_name):
        os.makedirs(dir_name)


def join(*args):
    args = [str(arg) for arg in args]
    return os.path.join(*args)


TYPE_DIR = 0
TYPE_FILE = 1
TYPE_ALL = 2


def listchildren(directory, children_type=TYPE_FILE, pattern=None, concat=False):
    path = Path(directory)
    children = list()
    for child in path.iterdir():
        if children_type == TYPE_ALL or (child.is_file() and children_type == TYPE_FILE) or\
                (child.is_dir() and children_type == TYPE_DIR):
            children.append(child)
    if pattern is not None and type(pattern) is str:
        children = [c for c in children if pu.search_pattern(pattern, c.name) is not None]
    children = [str(c) if concat else c.name for c in children]
    return sorted(children)
    # if children_type not in {TYPE_DIR, TYPE_FILE, TYPE_ALL}:
    #     print('listchildren() : Incorrect children type')
    #     return None
    # children = sorted(os.listdir(directory))
    # if pattern is not None:
    #     children = [c for c in children if pu.search_pattern(pattern, c) is not None]
    # if children_type != TYPE_ALL:
    #     res_list = list()
    #     for child in children:
    #         child_full_path = os.path.join(directory, child)
    #         if not os.path.exists(child_full_path):
    #             print('listchildren() : Invalid path')
    #             continue
    #         if children_type == TYPE_DIR and os.path.isdir(child_full_path) or \
    #                 (children_type == TYPE_FILE and os.path.isfile(child_full_path)):
    #             res_list.append(child)
    #     children = res_list
    # if concat:
    #     children = [os.path.join(directory, c) for c in children]
    # return children


def iterate_file_tree(root_path, func, *args, **kwargs):
    """
    Iterate file tree under the root path,, and invoke func in those directories with no sub-directories.
    :param root_path: Current location of tree iteration.
    :param func: A callback function.
    :param args: Just to pass over outer params for func.
    :param kwargs: Just to pass over outer params for func.
    """
    subdirs = listchildren(root_path, children_type=TYPE_DIR)
    if not subdirs:
        func(root_path, *args, **kwargs)
    else:
        for subdir in subdirs:
            iterate_file_tree(os.path.join(root_path, subdir), func, *args, **kwargs)
