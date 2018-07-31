#!/bin/bash

read RAW_TRAIN
read MODEL

mkdir -p tmp
mkdir -p ${MODEL}

# RAW_TRAIN="/home/nfs/cdong/tw/src/models/autophrase/raw_train.txt"
# MODEL="/home/nfs/cdong/tw/src/models/autophrase/default/"
# RAW_TRAIN is the input of AutoPhrase, where each line is a single document.

FIRST_RUN=${FIRST_RUN:- 1}
ENABLE_POS_TAGGING=${ENABLE_POS_TAGGING:- 1}
MIN_SUP=${MIN_SUP:- 5}
THREAD=${THREAD:- 10}

### Begin: Suggested Parameters ###
MAX_POSITIVES=-1
LABEL_METHOD=DPDN
RAW_LABEL_FILE=${RAW_LABEL_FILE:-""}


green=`tput setaf 2`
reset=`tput sgr0`


echo ${green}===Tokenization===${reset}
TOKENIZER="-cp .:tools/tokenizer/lib/*:tools/tokenizer/resources/:tools/tokenizer/build/ Tokenizer"
TOKENIZED_TRAIN=tmp/tokenized_train.txt
CASE=tmp/case_tokenized_train.txt
TOKEN_MAPPING=tmp/token_mapping.txt
if [ $FIRST_RUN -eq 1 ]; then
    echo -ne "Current step: Tokenizing input file...\033[0K\r"
    time java $TOKENIZER -m train -i $RAW_TRAIN -o $TOKENIZED_TRAIN -t $TOKEN_MAPPING -c N -thread $THREAD
fi
LANGUAGE='EN'
TOKENIZED_STOPWORDS=tmp/tokenized_stopwords.txt
TOKENIZED_ALL=tmp/tokenized_all.txt
TOKENIZED_QUALITY=tmp/tokenized_quality.txt
STOPWORDS=data/$LANGUAGE/stopwords.txt
ALL_WIKI_ENTITIES=data/$LANGUAGE/wiki_all.txt
QUALITY_WIKI_ENTITIES=data/$LANGUAGE/wiki_quality.txt
LABEL_FILE=tmp/labels.txt
if [ $FIRST_RUN -eq 1 ]; then
    echo -ne "Current step: Tokenizing stopword file...\033[0K\r"
    java $TOKENIZER -m test -i $STOPWORDS -o $TOKENIZED_STOPWORDS -t $TOKEN_MAPPING -c N -thread $THREAD
    echo -ne "Current step: Tokenizing wikipedia phrases...\033[0K\n"
    java $TOKENIZER -m test -i $ALL_WIKI_ENTITIES -o $TOKENIZED_ALL -t $TOKEN_MAPPING -c N -thread $THREAD
    java $TOKENIZER -m test -i $QUALITY_WIKI_ENTITIES -o $TOKENIZED_QUALITY -t $TOKEN_MAPPING -c N -thread $THREAD
fi


echo ${green}===Part-Of-Speech Tagging===${reset}
if [ ! $LANGUAGE == "JA" ] && [ ! $LANGUAGE == "CN" ]  && [ ! $LANGUAGE == "OTHER" ]  && [ $ENABLE_POS_TAGGING -eq 1 ] && [ $FIRST_RUN -eq 1 ]; then
    RAW=tmp/raw_tokenized_train.txt
    export THREAD LANGUAGE RAW
    bash ./tools/treetagger/pos_tag.sh
    mv tmp/pos_tags.txt tmp/pos_tags_tokenized_train.txt
fi


echo ${green}===AutoPhrasing===${reset}
if [ $ENABLE_POS_TAGGING -eq 1 ]; then
    time ./bin/segphrase_train \
        --pos_tag \
        --thread $THREAD \
        --pos_prune data/BAD_POS_TAGS.txt \
        --label_method $LABEL_METHOD \
		--label $LABEL_FILE \
        --max_positives $MAX_POSITIVES \
        --min_sup $MIN_SUP
else
    time ./bin/segphrase_train \
        --thread $THREAD \
        --label_method $LABEL_METHOD \
		--label $LABEL_FILE \
        --max_positives $MAX_POSITIVES \
        --min_sup $MIN_SUP
fi

echo ${green}===Generating Output===${reset}
java $TOKENIZER -m translate -i tmp/final_quality_salient.txt -o ${MODEL}/AutoPhrase.txt -t $TOKEN_MAPPING -c N -thread $THREAD
