import json
from os import listdir
from os.path import isfile, join, isdir
import traceback


def readFilesAsJsonList(fileDir):
    res_list = list()
    if type(fileDir) is list:
        try:
            for filepath in fileDir:
                res_list.extend(readFileAsJsonList(filepath))
        except:
            traceback.print_exc()
            print("read files error")
    elif isfile(fileDir):
        res_list = readFileAsJsonList(fileDir)
    elif isdir(fileDir):
        onlyfiles = [join(fileDir, f) for f in listdir(fileDir) if isfile(join(fileDir, f))]
        try:
            for filepath in onlyfiles:
                res_list.extend(readFileAsJsonList(filepath))
        except:
            traceback.print_exc()
            print("read files error")

    return res_list


def iterFileRead(filePath):
    with open(filePath, 'r') as fp:
        event_dict = json.load(fp)
    res = list()
    for key, value in event_dict.items():
        res.append(value)
    return res


def oneFileOneList(fileDir):
    onlyfiles = [join(fileDir, f) for f in listdir(fileDir) if isfile(join(fileDir, f))]
    res = list()
    for file_path in onlyfiles:
        res.append((file_path, readFileAsJsonList(file_path)))
    return res


def readFileAsJsonList(filePath):
    array = list()
    fp = open(filePath, 'r')
    try:
        for line in fp.readlines():
            try:
                array.append(json.loads(line.strip()))
            except:
                print(filePath)
                print(line)
                traceback.print_exc()
    except:
        traceback.print_exc()
        print("error while read", filePath)
    return array


def writeJsonListToFile(arrays, writeFilePath):
    with open(writeFilePath, mode='wt', encoding='utf-8') as myfile:
        for element in arrays:
            myfile.write(json.dumps(element) + '\n')
    # myfile.close


def get_all_substrings(input_string):
    length = len(input_string)
    return [input_string[i:j + 1] for i in range(length) for j in range(i, length)]
