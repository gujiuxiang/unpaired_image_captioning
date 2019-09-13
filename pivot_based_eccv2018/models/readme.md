NMTModel (
  (encoder): Encoder (
    (embeddings): Embeddings (
      (word_lut): Embedding(50004, 512, padding_idx=0)
      (dropout): Dropout (p = 0.5)
      (feature_luts): ModuleList (
      )
      (activation): ReLU ()
      (linear): BottleLinear (512 -> 512)
    )
    (rnn): LSTM(512, 256, dropout=0.5, bidirectional=True)
  )
  (decoder): Decoder (
    (embeddings): Embeddings (
      (word_lut): Embedding(50004, 512, padding_idx=0)
      (dropout): Dropout (p = 0.5)
      (feature_luts): ModuleList (
      )
      (activation): ReLU ()
      (linear): BottleLinear (512 -> 512)
    )
    (rnn): StackedLSTM (
      (dropout): Dropout (p = 0.5)
      (layers): ModuleList (
        (0): LSTMCell(1024, 512)
      )
    )
    (dropout): Dropout (p = 0.5)
    (attn): GlobalAttention (
      (linear_in): Linear (512 -> 512)
      (linear_out): Linear (1024 -> 512)
      (sm): Softmax ()
      (tanh): Tanh ()
    )
  )
  (generator): Sequential (
    (0): Linear (512 -> 50004)
    (1): LogSoftmax ()
  )
)
