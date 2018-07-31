import os
from subprocess import Popen, PIPE

import utils.function_utils as fu
import utils.tweet_keys as tk
import utils.timer_utils as tmu


autophrase_base = "/home/nfs/cdong/tw/src/tools/AutoPhrase"
model_base = autophrase_base + "/models/"
raw_train_file = os.path.join(model_base, "raw_train.txt")
output_file = os.path.join(model_base, "AutoPhrase.txt")
command = os.path.join(autophrase_base, "auto_phrase.sh")


def autophrase(raw, model):
    try:
        p = Popen(command, shell=True, bufsize=1, stdin=PIPE, stdout=PIPE, cwd=autophrase_base)
        p.stdin.flush()
        p.stdin.write((raw + '\n').encode("utf8"))
        p.stdin.write((model + '\n').encode("utf8"))
        p.communicate()
    except Exception as e:
        print('AutoPhrase Error', e)


def get_quality_phrase(twarr, threshold):
    quality_list = list()
    if len(twarr) == 0:
        return quality_list
    fu.write_lines(raw_train_file, [tw[tk.key_text] for tw in twarr])
    autophrase(raw_train_file, model_base)
    lines = fu.read_lines(output_file)
    for line in lines:
        confidence, phrase = line.strip().split(maxsplit=1)
        if float(confidence) > threshold:
            quality_list.append(phrase)
    return quality_list


if __name__ == "__main__":
    file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/2016-01-01_shoot_Aviv.json"
    twarr = fu.load_array(file)
    tmu.check_time()
    print(get_quality_phrase(twarr, 0.5))
    tmu.check_time()
