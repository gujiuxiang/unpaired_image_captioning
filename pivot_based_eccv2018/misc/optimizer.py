import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import misc.utils as utils


class Optim(object):
    def __init__(self, opt):
        self.last_ppl = None
        self.init_i2t(opt)
        self.init_nmt(opt)
        self._step = 0
        self.opt = opt

    def init_i2t(self, opt):
        self.i2t_train_flag = opt.i2t_train_flag
        self.i2t_eval_flag = opt.i2t_eval_flag
        self.i2t_method = opt.i2t_optim
        self.i2t_lr = opt.i2t_learning_rate
        self.i2t_current_lr = self.i2t_lr
        self.i2t_learning_rate_decay_start = opt.i2t_learning_rate_decay_start
        self.i2t_learning_rate_decay_every = opt.i2t_learning_rate_decay_every
        self.i2t_learning_rate_decay_rate = opt.i2t_learning_rate_decay_rate
        self.i2t_optim_alpha = opt.i2t_optim_alpha
        self.i2t_optim_beta = opt.i2t_optim_beta
        self.i2t_optim_epsilon = opt.i2t_optim_epsilon
        self.i2t_momentum = opt.i2t_momentum
        self.i2t_max_grad_norm = opt.i2t_max_grad_norm
        self.i2t_grad_clip = opt.i2t_grad_clip
        self.i2t_start_decay = False
        self.i2t_decay_method = opt.i2t_decay_method
        self.i2t_weight_decay = opt.i2t_weight_decay


    def init_nmt(self, opt):
        self.nmt_train_flag = opt.nmt_train_flag
        self.nmt_eval_flag = opt.nmt_eval_flag
        self.nmt_method = opt.nmt_optim
        self.nmt_lr = opt.nmt_learning_rate
        self.nmt_current_lr = self.nmt_lr
        self.nmt_learning_rate_decay_start = opt.nmt_learning_rate_decay_start
        self.nmt_learning_rate_decay_every = opt.nmt_learning_rate_decay_every
        self.nmt_learning_rate_decay_rate = opt.nmt_learning_rate_decay_rate
        self.nmt_optim_alpha = opt.nmt_optim_alpha
        self.nmt_optim_beta = opt.nmt_optim_beta
        self.nmt_optim_epsilon = opt.nmt_optim_epsilon
        self.nmt_momentum = opt.nmt_momentum
        self.nmt_max_grad_norm = opt.nmt_max_grad_norm
        self.nmt_grad_clip = opt.nmt_grad_clip
        self.nmt_start_decay = False
        self.nmt_decay_method = opt.nmt_decay_method
        self.nmt_weight_decay = opt.nmt_weight_decay

        self.nmt_warmup_steps = opt.nmt_warmup_steps
        self.nmt_betas = [0.9, 0.98]

    def create_optimizer(self, method, parameters, lr, alpha, beta, epsilon, weight_decay):
        if method == 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr, alpha, epsilon, weight_decay=weight_decay)
        elif method == 'adagrad':
            optimizer = optim.Adagrad(parameters, lr, weight_decay=weight_decay)
        elif method == 'sgd':
            optimizer = optim.SGD(parameters, lr, weight_decay=weight_decay)
        elif method == 'sgdm':
            optimizer = optim.SGD(parameters, lr, alpha, weight_decay=weight_decay)
        elif method == 'sgdmom':
            optimizer = optim.SGD(parameters, lr, alpha, weight_decay=weight_decay, nesterov=True)
        elif method == 'adam':
            optimizer = optim.Adam(parameters, lr, (alpha, beta), epsilon, weight_decay=weight_decay)
        else:
            raise RuntimeError("Invalid optim method: " + method)
        return optimizer

    def set_parameters(self, i2t_model, nmt_model):
        if i2t_model is not None:
            self.i2t_params = i2t_model.parameters()
            self.i2t_optimizer = self.create_optimizer(self.i2t_method, self.i2t_params, self.i2t_lr, self.i2t_optim_alpha, self.i2t_optim_beta, self.i2t_optim_epsilon, self.i2t_weight_decay)
            if vars(self.opt).get('start_from', None) is not None and os.path.isfile(os.path.join(self.opt.start_from, "i2t_optimizer.pth")):
                self.i2t_optimizer.load_state_dict(torch.load(os.path.join(self.opt.start_from, 'i2t_optimizer.pth')))

        if nmt_model is not None:
            self.nmt_params = list(nmt_model.parameters())  # careful: params may be a generator
            self.nmt_optimizer = self.create_optimizer(self.nmt_method, self.nmt_params, self.nmt_lr, self.nmt_optim_alpha, self.nmt_optim_beta, self.nmt_optim_epsilon, self.nmt_weight_decay)
            if vars(self.opt).get('start_from', None) is not None and os.path.isfile(os.path.join(self.opt.start_from, "nmt_optimizer.pth")):
                self.nmt_optimizer.load_state_dict(torch.load(os.path.join(self.opt.start_from, 'nmt_optimizer.pth')))

    def step(self):
        self._step += 1
        if self.i2t_train_flag:
            if self.i2t_max_grad_norm: clip_grad_norm(self.i2t_params, self.i2t_max_grad_norm)
            self.i2t_optimizer.step()
        if self.opt.nmt_train_flag:
            if self.opt.nmt_decay_method == "noam":
                self.nmt_current_lr = self.nmt_lr * (self.opt.rnn_size ** (-0.5) * min(self._step ** (-0.5), self._step * self.nmt_warmup_steps ** (-1.5)))
                for group in self.nmt_optimizer.param_groups:
                    group['lr'] = self.nmt_current_lr
            if self.nmt_max_grad_norm: clip_grad_norm(self.nmt_params, self.nmt_max_grad_norm)
            self.nmt_optimizer.step()

    def zero_grad(self):
        if self.i2t_train_flag:
            self.i2t_optimizer.zero_grad()
        if self.nmt_train_flag:
            self.nmt_optimizer.zero_grad()

    def update_ScheduledSampling_prob(self, opt, epoch, dp_i2t_model):
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            dp_i2t_model.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        return dp_i2t_model

    def update_LearningRate(self, type, epoch):
        if type == 'i2t':
            if epoch > self.i2t_learning_rate_decay_start and self.i2t_learning_rate_decay_start >= 0:
                frac = (epoch - self.i2t_learning_rate_decay_start) // self.i2t_learning_rate_decay_every
                decay_factor = self.i2t_learning_rate_decay_rate ** frac
                self.i2t_current_lr = self.i2t_lr * decay_factor
                for group in self.i2t_optimizer.param_groups:
                    group['lr'] = self.i2t_current_lr
            else:
                self.i2t_current_lr = self.i2t_lr

        if type == 'nmt':
            if epoch > self.nmt_learning_rate_decay_start and self.nmt_learning_rate_decay_start >= 0:
                self.nmt_current_lr = self.nmt_lr * self.nmt_learning_rate_decay_rate
                for group in self.nmt_optimizer.param_groups:
                    group['lr'] = self.nmt_current_lr
            else:
                self.nmt_current_lr = self.nmt_lr
