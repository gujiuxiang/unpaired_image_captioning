from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
from six.moves import cPickle
from torch.autograd import Variable
import torch.nn as nn
import opts
import misc.utils as utils
from trainer import *
from misc.dataloader.dataloader import *
from misc.dataloader.dataloader_coco import *

def init(opt):
    infos = {}
    histories = {}

    if opt.seed > 0: torch.manual_seed(opt.seed)
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5 # box means [prob, x,y,w,h]
    loader = DataLoader(opt)     # Load image captioning dataset
    coco_loader = DataLoader_COCO(opt) if opt.coco_eval_flag else None
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    if opt.start_from is not None and len(opt.start_from)>0:
        print('Start from: {}, Infos: {}'.format(opt.start_from, os.path.join(opt.start_from, 'infos-best.pkl')))
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos-best.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories-best.pkl')):
            with open(os.path.join(opt.start_from, 'histories-best.pkl')) as f:
                histories = cPickle.load(f)
    return opt, loader, coco_loader, infos, histories

def main(opt):
    opt, loader, coco_loader, infos, histories = init(opt)
    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    epoch_nmt = infos.get('epoch_nmt', 0)
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    loader.nmt_batchIdx = infos.get('nmt_batchIdx', loader.nmt_batchIdx)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    trainer = Trainer(opt, infos, loader, coco_loader)

    while True:
        start = time.time()
        data = loader.get_batch('train')  # Load data from train split (0)
        trainer.train(data, loader, iteration, epoch, epoch_nmt)
        add_summary_value(tf_summary_writer, 'read time/batch', time.time() - start, iteration)
        iteration += 1 # Update the iteration and epoch
        if data['bounds']['wrapped']:
            epoch += 1
            trainer.update_i2t_lr_flag = True
        if data['bounds']['wrapper_nmt']:
            epoch_nmt += 1
            trainer.update_nmt_lr_flag = True

        if (iteration % opt.losses_log_every == 0) and tf is not None:
            if opt.i2t_train_flag:
                loss_history[iteration] = trainer.i2t_train_loss if not trainer.sc_flag else trainer.i2t_avg_reward
                lr_history[iteration] = trainer.optim.i2t_current_lr
                ss_prob_history[iteration] = trainer.i2t_model.ss_prob
                add_summary_value(tf_summary_writer, 'i2t_train_loss', trainer.i2t_train_loss, iteration)
                add_summary_value(tf_summary_writer, 'i2t_learning_rate', trainer.optim.i2t_current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', trainer.i2t_model.ss_prob, iteration)
                add_summary_value(tf_summary_writer, 'i2t_avg_reward', trainer.i2t_avg_reward, iteration)

            if opt.nmt_train_flag:
                add_summary_value(tf_summary_writer, 'nmt_learning_rate', trainer.optim.nmt_current_lr, iteration)
                add_summary_value(tf_summary_writer, 'nmt_train_ppl', trainer.nmt_train_ppl, iteration)
                add_summary_value(tf_summary_writer, 'nmt_train_acc', trainer.nmt_train_acc, iteration)
                add_summary_value(tf_summary_writer, 'nmt_wemb_loss', trainer.wemb_loss, iteration)

            tf_summary_writer.flush()

        if (iteration % opt.save_checkpoint_every == 0) and tf is not None:
            trainer.eval(loader, coco_loader)
            if opt.nmt_eval_flag:
                add_summary_value(tf_summary_writer, 'nmt_valid_ppl', trainer.nmt_valid_ppl, iteration)
                add_summary_value(tf_summary_writer, 'nmt_valid_acc', trainer.nmt_valid_acc, iteration)
            if opt.i2t_eval_flag:
                add_summary_value(tf_summary_writer, 'validation loss', trainer.i2t_val_loss, iteration)
                for k, v in trainer.lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
            if opt.coco_eval_flag:
                for k, v in trainer.coco_lang_stats.items():
                    add_summary_value(tf_summary_writer, 'coco' + '_' + k, v, iteration)
            tf_summary_writer.flush()

            val_result_history[iteration] = {'loss': trainer.i2t_val_loss, 'lang_stats': trainer.lang_stats, 'coco_lang_stats': trainer.coco_lang_stats, 'predictions': trainer.predictions}
            val_result_history[iteration] = {'loss': trainer.i2t_val_loss, 'lang_stats': trainer.lang_stats, 'predictions': trainer.predictions}

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['epoch_nmt'] = epoch_nmt
            infos['iterators'] = loader.iterators
            infos['nmt_batchIdx'] = loader.nmt_batchIdx
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = trainer.best_i2t_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            infos['dicts'] = loader.nmt_dicts

            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history

            trainer.save_models()
            with open(os.path.join(opt.checkpoint_path, 'infos' + trainer.ckp_tag + '.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories' + trainer.ckp_tag + '.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

        i2t_data_processed = 100 * float(loader.iterators['train']) / float(len(loader.split_ix['train'])) if opt.i2t_train_flag else 0
        nmt_data_processed = 100 * float(loader.nmt_batchIdx) / float(len(loader.nmt_trainData)) if opt.nmt_train_flag else 0
        if opt.i2t_train_flag and opt.nmt_train_flag:
            print("{}|{}/{}/{}|I2T-{:.3f}%|NMT-{:.3f}%|TrainLoss:{:.2f}|AR:{:.2f}|TrainPPL:{:.2f}|TrainAcc:{:.2f}|WLoss:{:.4f}|TB:{:.2f}|" \
                .format(opt.id, iteration, epoch, epoch_nmt,
                        i2t_data_processed, nmt_data_processed,
                        trainer.i2t_train_loss,trainer.i2t_avg_reward,
                        trainer.nmt_train_ppl, trainer.nmt_train_acc, trainer.wemb_loss,
                        time.time() - start))
        elif opt.i2t_train_flag:
            print(
                "{}|{}/{}/{}|I2T-{:.3f}%|TrainLoss:{:.2f}|AR:{:.2f}|TB:{:.2f}|" \
                .format(opt.id, iteration, epoch, epoch_nmt,
                        i2t_data_processed,
                        trainer.i2t_train_loss, trainer.i2t_avg_reward,
                        time.time() - start))
        elif opt.nmt_train_flag:
            print(
                "{}|{}/{}/{}|NMT-{:.3f}%|TrainPPL:{:.2f}|TrainAcc:{:.2f}|WLoss:{:.4f}|TB:{:.2f}|" \
                .format(opt.id, iteration, epoch, epoch_nmt,
                        nmt_data_processed,
                        trainer.nmt_train_ppl, trainer.nmt_train_acc, trainer.wemb_loss,
                        time.time() - start))

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

if __name__ == "__main__":
    opt = opts.parse_opt()
    main(opt)
