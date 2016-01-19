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

### Training details
I set up 4 models:
* 3 layer Stacked LSTM
* 3 layer Grid LSTM
* 6 layer Stacked LSTM
* 6 layer Grid LSTM

Each model has 1000 units per layer, is trained with rmsprop, has an initial learning rate of 0.001, and uses the default weight decay settings from the char-rnn repo. I train on 50 length character sequences of batch size 50.

Each Grid LSTM has tied weights along the depth dimension as described in section 3.5. I also 'prioritize' the depth dimension of the network (section 3.2) by computing the transformed temporal hidden state prior to handing it to the depth LSTM.

I split the 100 million character dataset 95% training / 5% validation.

### Results
Here's the model's validation curves after running over the weekend:

![](https://github.com/coreylynch/grid-lstm/blob/master/val_curves.png)

> **NOTE**  
> Stacked LSTM w/ 6 layers flatlined w/ an average validation loss of 3.53 and is not shown.

Interestingly, I was basically unable to get the 6 layer stacked LSTM to learn anything (the validation loss flatlined at an average of 3.53 over the training period), whereas grid LSTM easily handled 6 layers. This was nice confirmation that the memory cells along the depth dimension do indeed help train significantly deeper networks. I also found it interesting that a 6 layer grid LSTM network converged to basically the same loss as a 3 layer grid LSTM network. I suspected the problem would benefit from additional depth, but I also trained without dropout so it might just need some additional regularization. I'll rerun and update.

### Cool related papers
There are a few contemporary architectures that provide similar gradient channeling along network depth: 
* [Gated Feedback Recurrent Neural Networks](http://arxiv.org/abs/1502.02367)
* [Depth-Gated LSTM](http://arxiv.org/abs/1508.03790)
* [Highway Networks](http://arxiv.org/abs/1505.00387): The authors of the grid LSTM paper actually point out that one dimensional grid LSTM (no temporal dimension) basically corresponds to a Highway Network: a feed-forward network that uses a gated linear transfer function in place of transfer functions such as tanh and ReLU.
