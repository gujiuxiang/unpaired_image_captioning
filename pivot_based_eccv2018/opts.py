import argparse
import datetime
import os
import sys

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i2t_train_flag', type=int, default=1, help='train i2t enable')
    parser.add_argument('--i2t_eval_flag', type=int, default=1, help='eval i2t enable')
    parser.add_argument('--nmt_train_flag', type=int, default=0, help='train nmt enable')
    parser.add_argument('--nmt_eval_flag', type=int, default=0, help='eval nmt enable')
    parser.add_argument('--coco_eval_flag', type=int, default=0, help='eval nmt enable')
    parser.add_argument('--nmt_kld_train_flag', type=int, default=0, help='train nmt enable')
    # Data input settings
    ## Image captioning part
    parser.add_argument('--use_blob_fetcher', type=int, default=1, help='switch, if set to 1, then use blob fetcher, otherwise, load h5 files')
    parser.add_argument('--input_fc_h5', type=str, default='data/aic_i2t/chinese_talk_1030_resnet101_fc.h5')
    parser.add_argument('--input_att_h5', type=str, default='data/aic_i2t/chinese_talk_1030_resnet101_att.h5')
    parser.add_argument('--input_fc_coco_h5', type=str, default='data/mscoco/cocotalk_karpathy_fc_0816.h5')
    parser.add_argument('--input_att_coco_h5', type=str, default='data/mscoco/cocotalk_karpathy_att_0816.h5')
    parser.add_argument('--input_json', type=str, default='data/aic_i2t/chinese_talk.json', help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_coco_json', type=str, default='data/mscoco/cocotalk_karpathy.json', help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/aic_i2t/bu_data/bu_fc', help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/aic_i2t/bu_data/bu_att', help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='data/aic_i2t/bu_data/bu_box', help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_box_cls_prob_dir', type=str, default='data/aic_i2t/bu_data/bu_box_cls_prob', help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_box_keep_boxes_dir', type=str, default='data/aic_i2t/bu_data/bu_box_keep_boxes',  help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/aic_i2t/chinese_talk_label.h5', help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_label_coco_h5', type=str, default='data/mscoco/cocotalk_karpathy_label_0816.h5', help='path to the h5file containing the preprocessed dataset')
    ## NMT part
    parser.add_argument('--input_nmt_choice', type=int, default=1, help='1:h5, 0:pt')
    parser.add_argument('--input_nmt_h5', default='data/aic_mt/processed/nmt.train.h5', help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('--input_nmt_pt', default='data/aic_mt/processed/nmt.train.pt', help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('--input_nmt_align', default='data/aic_mt/processed/nmt.alignments.pt', help='')
    parser.add_argument('--input_nmt_dict', default='data/aic_mt/processed/nmt.dicts.pt', help='')
    ## Start from
    parser.add_argument('--start_from', type=str, default='save/20180602-152023.transformer', help="continue training from saved model at this path")
    parser.add_argument('--cached_tokens', type=str, default='aic_i2t/chinese-train-idxs', help='Cached token file for calculating cider score during self critical training.')
    # Model settings
    ## Captioning model
    parser.add_argument('--caption_model', type=str, default="transformer", help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, stackcap, transformer')
    parser.add_argument('--rnn_size', type=int, default=1300, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512, help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512, help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--attri_hid_size', type=int, default=512, help='the hidden size of the attributes MLP')
    parser.add_argument('--fc_feat_size', type=int, default=2048, help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048, help='2048 for resnet, 512 for vgg')
    parser.add_argument('--attri_feat_size', type=int, default=1601, help='1601 for vg')
    parser.add_argument('--logit_layers', type=int, default=1, help='number of decode layers in the RNN')
    parser.add_argument('--use_bn', type=int, default=1, help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')
    ## NMT model
    parser.add_argument('--layers', type=int, default=1, help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('--word_vec_size', type=int, default=512, help='Word embedding sizes')
    parser.add_argument('--feature_vec_size', type=int, default=100, help='Feature vec sizes')
    parser.add_argument('--input_feed', type=int, default=1, help="""Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.""")
    parser.add_argument('--residual',   action="store_true", help="Add residual connections between RNN layers.")
    parser.add_argument('--brnn', action="store_true", default=True, help='Use a bidirectional encoder')
    parser.add_argument('--brnn_merge', default='concat', help="""Merge action for the bidirectional hidden states: [concat|sum]""")
    parser.add_argument('--copy_attn', action="store_true", help='Train copy attention layer.')
    parser.add_argument('--coverage_attn', action="store_true", help='Train a coverage attention layer.')
    parser.add_argument('--exhaustion_loss', action="store_true", help='')
    parser.add_argument('--lambda_exhaust', type=float, default=0.5, help='Lambda value for exhaustion.')
    parser.add_argument('--lambda_coverage', type=float, default=1, help='Lambda value for coverage.')
    parser.add_argument('--lambda_fertility', type=float, default=0.4, help='Lambda value for supervised fertility.')
    parser.add_argument('--encoder_layer', type=str, default='rnn', help="Type of encoder layer to use. Options: [rnn|mean|transformer]")
    parser.add_argument('--decoder_layer', type=str, default='rnn', help='Type of decoder layer to use. [rnn|transformer]')
    parser.add_argument('--context_gate', type=str, default=None, choices=['source', 'target', 'both'], help="""Type of context gate to use [source|target|both]. Do not select for no context gate.""")
    parser.add_argument('--attention_type', type=str, default='dotprod', choices=['dotprod', 'mlp'], help="""The attention type to use: dotprot (Luong) or MLP (Bahdanau)""")
    parser.add_argument('--attn_transform', type=str, default='softmax', choices=['softmax', 'constrained_softmax', 'sparsemax', 'constrained_sparsemax'], help="""The attention transform to use""")
    parser.add_argument('--c_attn', type=float, default=0.0, help="""c factor for increasing a by u""")
    parser.add_argument('--fertility', type=float, default=2.0, help="""Constant fertility value for each word in the source""")
    parser.add_argument('--predict_fertility', action="store_true", help="""Predict fertility value for each word in the source""")
    parser.add_argument('--guided_fertility', type=str, default=None, help="""Get fertility values from external aligner, specify alignment file""")
    parser.add_argument('--guided_fertility_source_file', type=str, default=None, help="""Get fertility values from external aligner, specify source file""")
    parser.add_argument('--supervised_fertility', type=str, default=None, help="""Get fertility values from external aligner, specify alignment file""")
    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=1, help='If normalize attention features')
    parser.add_argument('--use_box', type=int, default=1, help='If use box features')
    parser.add_argument('--use_box_cls_prob', type=int, default=1, help='If use box attributes probabilities?')
    parser.add_argument('--norm_box_feat', type=int, default=1, help='If use box, do we normalize box feature')
    # Optimization: General
    ## Captioning
    parser.add_argument('--max_epochs', type=int, default=-1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='minibatch size')
    parser.add_argument('--max_generator_batches', type=int, default=64, help="Maximum batches of words in a sequence to run the generator on in parallel.")
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=0, help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5, help='number of captions to sample for each image during training.')
    parser.add_argument('--beam_size', type=int, default=1, help='used when sample_max = 1, indicates number of beams in beam search.')
    ## Learning setting for image captioner
    parser.add_argument('--i2t_optim', type=str, default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--i2t_momentum', type=float, default=0)
    parser.add_argument('--i2t_learning_rate', type=float, default=4e-4, help="Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001")
    parser.add_argument('--i2t_learning_rate_decay_start', type=int, default=-1, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--i2t_learning_rate_decay_every', type=int, default=3, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--i2t_learning_rate_decay_rate', type=float, default=0.8, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--i2t_optim_alpha', type=float, default=0.9, help='alpha for adam')
    parser.add_argument('--i2t_optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--i2t_optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--i2t_decay_method', type=str, default="", help="""Use a custom learning rate decay [|noam] """)
    parser.add_argument('--i2t_weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--i2t_max_grad_norm', type=float, default=5, help="""If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('--i2t_grad_clip', type=float, default=0.1, help='clip gradients at this value')#5.,

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, help='Maximum scheduled sampling prob.')
    ## Learning setting for nmt
    parser.add_argument('--nmt_optim', type=str, default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--nmt_momentum', type=float, default=0)
    parser.add_argument('--nmt_learning_rate', type=float, default=1e-3, help="Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001")
    parser.add_argument('--nmt_learning_rate_decay_start', type=int, default=8, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--nmt_learning_rate_decay_every', type=int, default=3, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--nmt_learning_rate_decay_rate', type=float, default=0.5, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--nmt_optim_alpha', type=float, default=0.9, help='alpha for adam')
    parser.add_argument('--nmt_optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--nmt_optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--nmt_decay_method', type=str, default="", help="""Use a custom learning rate decay [|noam] """)
    parser.add_argument('--nmt_weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--nmt_max_grad_norm', type=float, default=5, help="""If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('--nmt_warmup_steps', type=int, default=4000, help="""Number of warmup steps for custom decay.""")
    parser.add_argument('--nmt_grad_clip', type=float, default=0.1, help='clip gradients at this value')#5.,

    #Optimization: for nmt
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('--position_encoding', action='store_true', help='Use a sinusoid to mark relative words positions.')
    parser.add_argument('--share_decoder_embeddings', action="store_true", help='Share the word and softmax embeddings for decoder.')
    parser.add_argument('--curriculum', action="store_true", help="""For this many epochs, order the minibatches based on source sequence length. Sometimes setting this to 1 will increase convergence speed.""")
    parser.add_argument('--extra_shuffle', action="store_true", help="""By default only shuffle mini-batch order; when true, shuffle and re-assign mini-batches""")
    parser.add_argument('--truncated_decoder', type=int, default=0, help="""Truncated bptt.""")

    # pretrained word vectors
    parser.add_argument('--pre_word_vecs_enc', help="If a valid path is specified, then load pretrained word embeddings on the encoder side.")
    parser.add_argument('--pre_word_vecs_dec', help="If a valid path is specified, then load pretrained word embeddings on the decoder side.")

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=100, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=100, help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='', help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1, help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--seed', type=int, default=-1, help="""Random seed used for the experiments reproducibility.""")
    parser.add_argument('--id', type=str, default='', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0, help='if true then use 80k, else use 110k')
    parser.add_argument('--gpus', default=[0], nargs='+', type=int, help="Use CUDA on the listed devices.")
    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1, help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0, help='The reward weight from bleu4')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    if len(args.checkpoint_path) >0:
        args.id = args.checkpoint_path.replace('save/', '').replace('rl/', '')
        print('Model ID: {}'.format(args.id))
    if len(args.id) == 0:
        model_tag = args.caption_model + '_nmt' if args.nmt_train_flag else args.caption_model
        args.checkpoint_path = 'save/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.' + model_tag
        print('New Checkpoint path is : {}'.format(args.checkpoint_path))
        args.id = args.checkpoint_path.replace('save/','')

    return args
