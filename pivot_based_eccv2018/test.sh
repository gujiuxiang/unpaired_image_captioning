#!/usr/bin/env bash
clear
func_nmt_eval()
{
    echo "Start machine translation ..."
    SOURCE_DIR=$PWD"/../OpenNMT-py"
    eval cd "${SOURCE_DIR}"
    export PYTHONPATH="$PYTHONPATH:$SOURCE_DIR"
    echo "Start from source dir: "$SOURCE_DIR
    echo "Model:"NMT_MODEL_NAME "Src:"$SRC_FILE "Tgt:"$TGT_FILE
    python translate.py -model $NMT_MODEL_NAME -src $SRC_FILE -output $TGT_FILE -verbose -gpu 0
}

func_i2t_pivot_eval()
{
    echo "Start image caption generation ..."
    SOURCE_DIR=$PWD
    eval cd "${SOURCE_DIR}"
    export PYTHONPATH="$PYTHONPATH:$SOURCE_DIR"
    echo "Start from source dir: "$SOURCE_DIR
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_pivot.py       --model $I2T_MODEL_NAME \
                                                            --dataset aic_i2c \
                                                            --split 'test' \
                                                            --batch_size 50 \
                                                            --num_images 5000 \
                                                            --sample_max 1 \
                                                            --max_ppl 0 \
                                                            --image_folder "" \
                                                            --input_json "data/mscoco/cocotalk_karpathy.json" \
                                                            --input_fc_dir "data/mscoco/bu_data/bu_fc" \
                                                            --input_att_dir "data/mscoco/bu_data/bu_att" \
                                                            --input_box_dir "data/mscoco/bu_data/bu_box" \
                                                            --input_box_cls_prob_dir "data/mscoco/bu_data/bu_box_cls_prob" \
                                                            --input_box_keep_boxes_dir "data/mscoco/bu_data/bu_box_keep_boxes" \
                                                            --input_label_h5 "data/mscoco/cocotalk_karpathy_label.h5"
}

func_i2t_paired_eval()
{
    echo "Start image caption generation ..."
    SOURCE_DIR=$PWD
    eval cd "${SOURCE_DIR}"
    export PYTHONPATH="$PYTHONPATH:$SOURCE_DIR"
    echo "Start from source dir: "$SOURCE_DIR
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_paired.py      --model $I2T_MODEL_NAME \
                                                            --dataset aic_i2c \
                                                            --split 'test' \
                                                            --batch_size 50 \
                                                            --beam_size 5 \
                                                            --num_images 10000 \
                                                            --sample_max 1 \
                                                            --max_ppl 0 \
                                                            --image_folder "" \
                                                            --input_json "data/aic_i2t/chinese_talk.json" \
                                                            --input_fc_dir "data/aic_i2t/bu_data/bu_fc" \
                                                            --input_att_dir "data/aic_i2t/bu_data/bu_att" \
                                                            --input_box_dir "data/aic_i2t/bu_data/bu_box" \
                                                            --input_box_cls_prob_dir "data/aic_i2t/bu_data/bu_box_cls_prob" \
                                                            --input_box_keep_boxes_dir "data/aic_i2t/bu_data/bu_box_keep_boxes" \
                                                            --input_label_h5 "data/aic_i2t/chinese_talk_label.h5"
}

func_style_eval()
{
    echo "Start language style transfering ..."
    SOURCE_DIR=$PWD/misc/language-style-transfer/code
    eval cd "${SOURCE_DIR}"
    export PYTHONPATH="$PYTHONPATH:$SOURCE_DIR"
    echo "Start from source dir: "$SOURCE_DIR
    python style_transfer.py --test /home/jxgu/github/unparied_im2text_jxgu/tmp/coco_test_5k_en.txt --output /home/jxgu/github/unparied_im2text_jxgu/tmp/sentiment.test --vocab ../tmp/captions.vocab --model ../tmp/model-captions --load_model true
}

GPU_ID=$1 # Get gpu id
ROOT_DIR=$PWD
echo "Set root dir to: ""$ROOT_DIR"
IAMGE_FOLDER="$ROOT_DIR/../im2text_jxgu/pytorch/data/mscoco"
COCO_JSON="$ROOT_DIR/../github/im2text_jxgu/pytorch/data/mscoco/image_info_karpathy_5k_test.json"
START_FROM="$ROOT_DIR/save/20180208-063759.fcnmt_True"
NMT_MODEL_NAME="save/opennmt/aic_zh2en_nmt_part_acc_60.06_ppl_6.83_e23.pt"
I2T_MODEL_NAME="save/20180602-152023.transformer/model_i2t-best.pth"
SRC_FILE="$ROOT_DIR/tmp/coco_test_5k_zh_bak2.txt"
TGT_FILE="$ROOT_DIR/tmp/coco_test_5k_zh.en.txt"

#func_nmt_eval
func_i2t_pivot_eval
#func_i2t_paired_eval
#func_style_eval