#! /bin/sh

clear
#-----------------------------------------------------------------------------------------------------------------------
func_i2t_xe_rl()
{
    #-----------------------------------------
    XE_ENABLE=1
    if [ "$XE_ENABLE" -eq 1 ];then
        TIME_TAG=`date "+%Y%m%d-%H%M%S"` # Time stamp
        if [ -z "$CKPT_PATH" ]; then
            CKPT_PATH="save/"
        fi
        echo "Current saving path is "$CKPT_PATH
        MODEL_ID=${CKPT_PATH#"save/"}
        echo "Current saving id is "$MODEL_ID
        if [ ! -f $CKPT_PATH"/infos-best.pkl" ]; then
            START_FROM=""
        else
            START_FROM="--start_from "$CKPT_PATH
        fi
        echo "Current checkpoint path: "$CKPT_PATH
        echo "Start from pretrained: "$START_FROM

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --caption_model $MODEL_TYPE \
                                                        --i2t_train_flag 1 \
                                                        --i2t_eval_flag 1 \
                                                        --nmt_train_flag 0 \
                                                        --nmt_eval_flag 0 \
                                                        --batch_size 50 \
                                                        --beam_size 1 \
                                                        --i2t_learning_rate 5e-4 \
                                                        --i2t_learning_rate_decay_start 0 \
                                                        --scheduled_sampling_start 0 \
                                                        --checkpoint_path $CKPT_PATH $START_FROM \
                                                        --save_checkpoint_every 1000 \
                                                        --language_eval 1 \
                                                        --val_images_use 10000 \
                                                        --max_epoch 100  \
                                                        --self_critical_after 37 \
                                                        --rnn_size 1300 \
                                                        --use_box 1 \
                                                        --use_bn 1 \
                                                        --use_box_cls_prob 1 \
                                                        --norm_box_feat 1 \
                                                        --gpus $(echo $GPU_ID | tr "," "\n") | tee $CKPT_PATH/log_train_$TIME_TAG.txt
    fi
    #-----------------------------------------
    RL_ENABLE=1
    if [ "$RL_ENABLE" -eq 1 ];then
        echo "Current saving path is "$CKPT_PATH
        MODEL_ID=${CKPT_PATH#"save/"}
        echo "Current saving id is "$MODEL_ID
        if [ ! -f $CKPT_PATH"/infos-best.pkl" ]; then
            START_FROM=""
        else
            START_FROM="--start_from "$CKPT_PATH
        fi
        echo "Current checkpoint path: "$CKPT_PATH
        echo "Start from pretrained: "$START_FROM
        if [ ! -d save/rl ]; then
            mkdir save/rl
        fi
        if [ ! -d save/rl/$MODEL_ID ]; then
            cp -r $CKPT_PATH save/xe/
        fi
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --caption_model $MODEL_TYPE \
                                                        --i2t_train_flag 1 \
                                                        --i2t_eval_flag 1 \
                                                        --nmt_train_flag 0 \
                                                        --nmt_eval_flag 0 \
                                                        --batch_size 20 \
                                                        --beam_size 1 \
                                                        --i2t_learning_rate 5e-5 \
                                                        --i2t_learning_rate_decay_start 0 \
                                                        --i2t_learning_rate_decay_every 55  \
                                                        --i2t_learning_rate_decay_rate 0.1  \
                                                        --scheduled_sampling_start 0 \
                                                        --checkpoint_path save/rl/$MODEL_ID \
                                                        --start_from $CKPT_PATH \
                                                        --save_checkpoint_every 1000 \
                                                        --language_eval 1 \
                                                        --val_images_use 10000 \
                                                        --self_critical_after 37 \
                                                        --rnn_size 1300 \
                                                        --use_bn 1 \
                                                        --use_box 1 \
                                                        --use_box_cls_prob 1 \
                                                        --norm_box_feat 1 \
                                                        --gpus $(echo $GPU_ID | tr "," "\n") | tee $CKPT_PATH/log_train_$TIME_TAG.txt
    fi
}

func_nmt_xe_rl()
{
    #-----------------------------------------
    XE_ENABLE=1
    if [ "$XE_ENABLE" -eq 1 ];then
        TIME_TAG=`date "+%Y%m%d-%H%M%S"` # Time stamp
        if [ -z "$CKPT_PATH" ]; then
            CKPT_PATH="save/"
        fi
        echo "Current saving path is "$CKPT_PATH
        MODEL_ID=${CKPT_PATH#"save/"}
        echo "Current saving id is "$MODEL_ID
        if [ ! -f $CKPT_PATH"/infos-best.pkl" ]; then
            START_FROM=""
        else
            START_FROM="--start_from "$CKPT_PATH
        fi
        echo "Current checkpoint path: "$CKPT_PATH
        echo "Start from pretrained: "$START_FROM

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --caption_model $MODEL_TYPE \
                                                        --i2t_train_flag 0 \
                                                        --i2t_eval_flag 0 \
                                                        --nmt_train_flag 1 \
                                                        --nmt_eval_flag 1 \
                                                        --batch_size 64 \
                                                        --beam_size 1 \
                                                        --checkpoint_path $CKPT_PATH $START_FROM \
                                                        --save_checkpoint_every 1000 \
                                                        --max_epoch 100 \
                                                        --rnn_size 1300 \
                                                        --use_box 1 \
                                                        --use_bn 1 \
                                                        --use_box_cls_prob 1 \
                                                        --norm_box_feat 1 \
                                                        --gpus $(echo $GPU_ID | tr "," "\n") | tee $CKPT_PATH/log_train_$TIME_TAG.txt
    fi
}


func_nmt_style()
{
    PROCESS_DIR="misc/language-style-transfer/code"
    echo $PWD/$PROCESS_DIR
    eval cd "${PWD}/$PROCESS_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID python style_transfer.py   --train ../data/aic_coco/captions.train \
                                                            --dev ../data/aic_coco/captions.dev \
                                                            --output ../tmp/captions.dev \
                                                            --vocab ../tmp/captions.vocab \
                                                            --load_model ../tmp/model-captions \
                                                            --model ../tmp/model-captions
}

func_nmt_offical_zh2en()
{
    FRAMEWORK=1
    TIME_TAG=`date "+%Y%m%d-%H%M%S"` # Time stamp
    PROCESS_DIR="${PWD}/../OpenNMT-py"
    DATA_DIR="${PWD}/data/aic_mt/processed"
    echo $PROCESS_DIR
    eval cd "$PROCESS_DIR"
    if [ ${#CKPT_PATH} -ge 1 ]; then
        echo "Training from pretrained" ;
        case "$FRAMEWORK" in
             0) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_zh2en_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_zh2en_nmt_part \
                                                             -epochs 50 \
                                                             -batch_size 128 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             -train_from ${ROOT_DIR}/save/opennmt/$CKPT_PATH \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             1) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_zh2en_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_zh2en_nmt_part \
                                                             -layers 6 -rnn_size 512 -word_vec_size 512   \
                                                             -encoder_type transformer -decoder_type transformer -position_encoding \
                                                             -epochs 50  -max_generator_batches 32 -dropout 0.1 \
                                                             -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
                                                             -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
                                                             -max_grad_norm 0 -param_init 0  -param_init_glorot \
                                                             -label_smoothing 0.1 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             -train_from ${ROOT_DIR}/save/opennmt/$CKPT_PATH \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             *) echo "No input" ;;
        esac
    else
        case "$FRAMEWORK" in
             0) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_zh2en_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_zh2en_nmt_part \
                                                             -epochs 50 \
                                                             -batch_size 128 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             1) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_zh2en_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_zh2en_nmt_part \
                                                             -layers 6 -rnn_size 512 -word_vec_size 512   \
                                                             -encoder_type transformer -decoder_type transformer -position_encoding \
                                                             -epochs 50  -max_generator_batches 32 -dropout 0.1 \
                                                             -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
                                                             -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
                                                             -max_grad_norm 0 -param_init 0  -param_init_glorot \
                                                             -label_smoothing 0.1 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             *) echo "No input" ;;
        esac
    fi
}

func_nmt_offical_en2zh()
{
    FRAMEWORK=1
    TIME_TAG=`date "+%Y%m%d-%H%M%S"` # Time stamp
    PROCESS_DIR="${PWD}/../OpenNMT-py"
    DATA_DIR="${PWD}/data/aic_mt/processed"
    echo $PROCESS_DIR
    eval cd "$PROCESS_DIR"
    if [ ${#CKPT_PATH} -ge 1 ]; then
        echo "Training from pretrained" ;
        case "$FRAMEWORK" in
             0) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_en2zh_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_en2zh_nmt_part \
                                                             -epochs 50 \
                                                             -batch_size 128 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             -train_from ${ROOT_DIR}/save/opennmt/$CKPT_PATH \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             1) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_en2zh_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_en2zh_nmt_part \
                                                             -layers 6 -rnn_size 512 -word_vec_size 512   \
                                                             -encoder_type transformer -decoder_type transformer -position_encoding \
                                                             -epochs 50  -max_generator_batches 32 -dropout 0.1 \
                                                             -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
                                                             -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
                                                             -max_grad_norm 0 -param_init 0  -param_init_glorot \
                                                             -label_smoothing 0.1 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             -train_from ${ROOT_DIR}/save/opennmt/$CKPT_PATH \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             *) echo "No input" ;;
        esac
    else
        case "$FRAMEWORK" in
             0) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_en2zh_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_en2zh_nmt_part \
                                                             -epochs 50 \
                                                             -batch_size 128 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             1) CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -data $DATA_DIR/nmt_en2zh_part \
                                                             -save_model ${ROOT_DIR}/save/opennmt/aic_en2zh_nmt_part \
                                                             -layers 6 -rnn_size 512 -word_vec_size 512   \
                                                             -encoder_type transformer -decoder_type transformer -position_encoding \
                                                             -epochs 50  -max_generator_batches 32 -dropout 0.1 \
                                                             -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
                                                             -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
                                                             -max_grad_norm 0 -param_init 0  -param_init_glorot \
                                                             -label_smoothing 0.1 \
                                                             -gpu 1 \
                                                             -gpuid 0 \
                                                             | tee log_nmt_train_$TIME_TAG.txt;;
             *) echo "No input" ;;
        esac
    fi
}

func_babytalk()
{
    PROCESS_DIR="${PWD}/misc/variants/NeuralBabyTalk"
    echo $PROCESS_DIR
    eval cd "$PROCESS_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 100 --cuda True --num_workers 1 --max_epoch 30 --mGPUs Ture
}

MODEL_TYPE=''
ROOT_DIR=$PWD
echo "run "$MODEL_TYPE
GPU_ID=$2 # Get gpu id
CKPT_PATH=$3 # Get save path
echo "GPU using "$GPU_ID

case "$1" in
     0) MODEL_TYPE='denseatt' && func_i2t_xe_rl;;
     1) MODEL_TYPE='stackcap' && func_i2t_xe_rl;;
     2) MODEL_TYPE='transformer' && func_i2t_xe_rl;;
     3) MODEL_TYPE='fc' && func_i2t_xe_rl;;
     4) MODEL_TYPE='na' && func_nmt_xe_rl;;
     5) func_nmt_offical_zh2en;;
     6) func_nmt_offical_en2zh;;
     7) MODEL_TYPE='nmt_style' && func_nmt_style;;
     8) func_babytalk;;
     *) echo "No input" ;;
esac