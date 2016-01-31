# 2D Grid LSTM

![](https://github.com/coreylynch/grid-lstm/blob/master/grid-lstm.png)

This is a Torch 7 implementation of 2D grid LSTM described here: http://arxiv.org/pdf/1507.01526v2.pdf. See model/GridLSTM.lua for the implementation. The rest of the code (modulo some small changes in train.lua) come from [karpathy's character level rnn repo](https://github.com/karpathy/char-rnn). See that repo for installation and usage details. This basically just allows you to call
```
th train.lua -model grid_lstm
```

### Intro
2D Grid LSTM differs from traditional stacked LSTM by adding LSTM cells along the depth dimension of the network as well as the temporal dimension. That is, each layer uses both a hidden state and a memory cell to communicate to the next. This gives the depth dimension the same gradient channeling properties available along the temporal dimension, helping to mitigate the vanishing gradient problem in networks with many layers and allowing layers to dynamically select or ignore their inputs. 

### Small Experiment on Wikipedia
Since the promise of 2D grid LSTMs seems to be the ability to train deeper recurrent nets, I wanted to see the advantages in action on a dataset complex enough to warrant some depth, so I ran a small experiment similar to section 4.3 of the paper: character-level language modeling on the 100m character [Hutter challenge](http://prize.hutter1.net/) Wikipedia dataset. (For an actual evaluation of these models, see the original paper; this was just a sanity check.)

#### Training details
I set up 4 models:
* 3 layer Stacked LSTM
* 3 layer Grid LSTM
* 6 layer Stacked LSTM
* 6 layer Grid LSTM

Each model has 1000 units per layer, is trained with rmsprop, has an initial learning rate of 0.001, and uses the default weight decay and gradient clipping settings from the char-rnn repo. I train on 50 length character sequences of batch size 50. I add 25% dropout on the non-recurrent connections to all models.

Each Grid LSTM has tied weights along the depth dimension as described in section 3.5. I also 'prioritize' the depth dimension of the network (section 3.2) by computing the transformed temporal hidden state prior to handing it to the depth LSTM.

I split the 100 million character dataset 95% training / 5% validation.

#### Results
Here are each models' validation curves:

![](https://github.com/coreylynch/grid-lstm/blob/master/val_curves.png)

It's clear from the graph that grid LSTM converges to a better model than stacked LSTM on this task. The best grid LSTM network (6 layers) has a 4.61% lower validation loss than the best stacked LSTM (3 layers). This was nice confirmation that the linear LSTM gating mechanism along the depth dimension does indeed help when training deeper recurrent networks. 

I was trying to reason for myself about why this kind of linear information flow along the depth of the network might be so beneficial. One story I could imagine goes like this:

Say a memory cell in a lower layer in the network activates when inside a URL. [Karpathy and Johnson](http://arxiv.org/pdf/1506.02078v2.pdf) actually find many concrete examples of character language model LSTMs using their memory cells to remember long-range information just like this, like cells that activate inside quotes, inside comments, with increasing strength relative to line position, etc. Let’s also suppose that this "am I inside a URL?" memory cell's current activation value is relevant to a higher layer’s processing. 

In traditional stacked LSTM, this information in the lower cell has to travel through an output gate, a tanh nonlinearity, an input gate and another tanh nonlinearity to reach the upper cell. A grid LSTM network, on the other hand, can write the information to a lower cell, close the forget gate on it carrying it across multiple layers, then expose the information directly to some higher layer. This replaces the prior path through multiple non-linearities with a linear identity transformation, modulated by a forget gate. 

I could see how this ability to pass information unchanged through many layers might improve credit assignment and make training easier. The authors of [Highway Networks](http://arxiv.org/abs/1505.00387) also make a convincing case for how this kind of gating mechanism between layers makes information flow more efficiently through a trained network. 

#### Note on Dropout
Dropout was necessary for getting the best performance out of both traditional stacked LSTMs and grid LSTMs for this task. Interestingly, without dropout I was unable to train a 6 layer stacked LSTM on this dataset (the validation loss flatlined over the training period with an average loss of 3.53), whereas I was able to train a 6 layer grid LSTM easily with no dropout. For more regularizing LSTMs with dropout, see [this](http://arxiv.org/abs/1409.2329).

### Cool related papers
There are a few contemporary architectures that provide similar gradient channeling along network depth: 
* [Gated Feedback Recurrent Neural Networks](http://arxiv.org/abs/1502.02367)
* [Depth-Gated LSTM](http://arxiv.org/abs/1508.03790)
* [Highway Networks](http://arxiv.org/abs/1505.00387): The authors of the grid LSTM paper actually point out that one dimensional grid LSTM (no temporal dimension) basically corresponds to a Highway Network: a feed-forward network that uses a gated linear transfer function in place of transfer functions such as tanh and ReLU.
