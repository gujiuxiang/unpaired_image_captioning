#!/usr/bin/env bash

func_bottom_up_feature_extract()
{
    PROCESS_DIR=~/github/bottom-up-attention/tools
    eval cd "${PROCESS_DIR}"
    python generate_tsv.py --gpu 2,4 --def ../models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net ../data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --out karpathy_train_resnet101_faster_rcnn_genome --cfg ../experiments/cfgs/faster_rcnn_end2end_resnet.yml --split coco_val2014
}


func_opennmt_process_zh2en()
{
    PROCESS_DIR=$PWD
    eval cd "${PROCESS_DIR}"
    export PYTHONPATH="$PYTHONPATH:$PROCESS_DIR"
    echo "Start from source dir: "$PROCESS_DIR
    python scripts/prepro_aic_nmt.py    -train_src $DATA_DIR/train.zh \
                                        -train_tgt $DATA_DIR/train.en \
                                        -valid_src $DATA_DIR/valid.zh \
                                        -valid_tgt $DATA_DIR/valid.en \
                                        -save_data $DATA_DIR/nmt_zh2en
}

func_opennmt_process_en2zh()
{
    PROCESS_DIR=$PWD
    eval cd "${PROCESS_DIR}"
    export PYTHONPATH="$PYTHONPATH:$PROCESS_DIR"
    echo "Start from source dir: "$PROCESS_DIR
    python scripts/prepro_aic_nmt.py    -train_src $DATA_DIR/train.en \
                                        -train_tgt $DATA_DIR/train.zh \
                                        -valid_src $DATA_DIR/valid.en \
                                        -valid_tgt $DATA_DIR/valid.zh \
                                        -save_data $DATA_DIR/nmt_en2zh
}

func_opennmt_offical_process_zh2en()
{
    PROCESS_DIR=~/github/OpenNMT-py
    eval cd "${PROCESS_DIR}"
    export PYTHONPATH="$PYTHONPATH:$PROCESS_DIR"
    echo "Start from source dir: "$PROCESS_DIR
    python preprocess.py    -train_src $DATA_DIR/train.zh \
                            -train_tgt $DATA_DIR/train.en \
                            -valid_src $DATA_DIR/valid.zh \
                            -valid_tgt $DATA_DIR/valid.en \
                            -max_shard_size "$((50 * 1024 * 1024))" \
                            -save_data $DATA_DIR/nmt_zh2en_part
}

func_opennmt_offical_process_en2zh()
{
    PROCESS_DIR=~/github/OpenNMT-py
    eval cd "${PROCESS_DIR}"
    export PYTHONPATH="$PYTHONPATH:$PROCESS_DIR"
    echo "Start from source dir: "$PROCESS_DIR
    python preprocess.py    -train_src $DATA_DIR/train.en \
                            -train_tgt $DATA_DIR/train.zh \
                            -valid_src $DATA_DIR/valid.en \
                            -valid_tgt $DATA_DIR/valid.zh \
                            -max_shard_size "$((50 * 1024 * 1024))" \
                            -save_data $DATA_DIR/nmt_en2zh_part
}

func_preprocess()
{
    eval cd "${AIC_ROOT}"
    #unwrap xml for valid data and test data
    python prepare_data/unwrap_xml.py $TMP_DIR/translation_validation_20170912/valid.en-zh.zh.sgm >$DATA_DIR/valid.en-zh.zh
    python prepare_data/unwrap_xml.py $TMP_DIR/translation_validation_20170912/valid.en-zh.en.sgm >$DATA_DIR/valid.en-zh.en

    #Prepare Data

    ##Chinese words segmentation
    python prepare_data/jieba_cws.py $TMP_DIR/translation_train_20170912/train.zh > $DATA_DIR/train.zh
    python prepare_data/jieba_cws.py $DATA_DIR/valid.en-zh.zh > $DATA_DIR/valid.zh
    ## Tokenize and Lowercase English training data
    cat $TMP_DIR/translation_train_20170912/train.en | prepare_data/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/train.en
    cat $DATA_DIR/valid.en-zh.en | prepare_data/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/valid.en

    #Bulid Dictionary
    python prepare_data/build_dictionary.py $DATA_DIR/train.en
    python prepare_data/build_dictionary.py $DATA_DIR/train.zh

    src_vocab_size=50000
    trg_vocab_size=50000
    python prepare_data/generate_vocab_from_json.py $DATA_DIR/train.en.json ${src_vocab_size} > $DATA_DIR/vocab.en
    python prepare_data/generate_vocab_from_json.py $DATA_DIR/train.zh.json ${trg_vocab_size} > $DATA_DIR/vocab.zh

    #rm -r $DATA_DIR/train.*.json
}

clear
ROOT_DIR=$PWD
AIC_ROOT=~/github/AI_Challenger/Baselines/translation_and_interpretation_baseline/train
DATA_DIR=$PWD/data/aic_mt/processed
TMP_DIR=~/github/unparied_im2text_jxgu/data/aic_mt/raw_data
mkdir -p $DATA_DIR

#func_preprocess
#func_opennmt_process_zh2en
#func_opennmt_process_en2zh
#func_opennmt_offical_process_zh2en
#func_opennmt_offical_process_en2zh
func_bottom_up_feature_extract
