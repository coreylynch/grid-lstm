# 2D Grid LSTM

This is a Torch 7 implementation of 2D grid LSTM described here: http://arxiv.org/pdf/1507.01526v2.pdf. See model/GridLSTM.lua for the implementation. The rest of the code (modulo some small changes in train.lua) come from [karpathy's character level rnn repo](https://github.com/karpathy/char-rnn)

2D Grid LSTM differs from traditional stacked LSTM by adding memory cells along the depth dimension of the network as well as the temporal dimension. That is, each layer uses both a hidden state and a memory cell to communicate to the next. This gives the depth dimension the same gradient channeling properties available along the temporal dimension, helping to mitigate the vanishing gradient problem in networks with many layers and allowing layers to dynamically select or ignore their inputs.

