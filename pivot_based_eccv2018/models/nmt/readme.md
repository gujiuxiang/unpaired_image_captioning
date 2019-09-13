attention_type='dotprod', 
attn_transform='softmax', 
batch_size=128, 
brnn=True, 
brnn_merge='concat', 
c_attn=0.0, 
context_gate=None, 
copy_attn=False,
coverage_attn=False,
curriculum=False,
data='data/ai_challenger/machine_translation/nmt_t2t_data_1k/nmt_1k.train.pt', decay_method='', decoder_layer='rnn',
dropout=0.3,
encoder_layer='rnn',
encoder_type='text',
epochs=100,
exhaustion_loss=False,
experiment_name='',
extra_shuffle=False,
feature_vec_size=100,
fertility=2.0, gpus=[0],
guided_fertility=None,
guided_fertility_source_file=None,
input_feed=1,
lambda_coverage=1,
lambda_exhaust=0.5,
lambda_fertility=0.4,
layers=1,
learning_rate=1.0,
learning_rate_decay=0.5,
log_interval=50,
log_server='',
max_generator_batches=32,
max_grad_norm=5,
momentum=0,
optim='sgd', 
param_init=0.1,
position_encoding=False, 
pre_word_vecs_dec=None, 
pre_word_vecs_enc=None, 
predict_fertility=False, 
rnn_size=512, 
rnn_type='LSTM', 
save_model='save/demo-model-0210-full', 
seed=-1, 
share_decoder_embeddings=False, 
start_checkpoint_at=0, 
start_decay_at=8, 
start_epoch=1, 
supervised_fertility=None, 
train_from='', 
train_from_state_dict='', 
truncated_decoder=0, 
warmup_steps=4000, 
word_vec_size=512

 * vocabulary size. source = 11986; target = 8571
 * number of training sentences. 10000
 * maximum batch size. 128
Building model...
Intializing params
NMTModel (
  (encoder): Encoder (
    (embeddings): Embeddings (
      (word_lut): Embedding(11986, 512, padding_idx=0)
      (dropout): Dropout (p = 0.3)
      (feature_luts): ModuleList (
      )
      (activation): ReLU ()
      (linear): BottleLinear (512 -> 512)
    )
    (rnn): LSTM(512, 256, dropout=0.3, bidirectional=True)
  )
  (decoder): Decoder (
    (embeddings): Embeddings (
      (word_lut): Embedding(8571, 512, padding_idx=0)
      (dropout): Dropout (p = 0.3)
      (feature_luts): ModuleList (
      )
    )
    (rnn): StackedLSTM (
      (dropout): Dropout (p = 0.3)
      (layers): ModuleList (
        (0): LSTMCell(1024, 512)
      )
    )
    (dropout): Dropout (p = 0.3)
    (attn): GlobalAttention (
      (linear_in): Linear (512 -> 512)
      (linear_out): Linear (1024 -> 512)
      (sm): Softmax ()
      (tanh): Tanh ()
    )
  )
  (generator): Sequential (
    (0): Linear (512 -> 8571)
    (1): LogSoftmax ()
  )
)
* number of parameters: 20697979