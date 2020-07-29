Char-RNN implementation inspired by [Karpathy's char-rnn](https://github.com/karpathy/char-rnn).

### Usage:

To train with default parameters on the Shakespeare corpus, run `python train.py`. To access all the parameters use `python train.py --help`.
To sample from a checkpointed model, `python sample.py`. 
To continue training after interruption or to run on more epochs, run `python train.py --init_from=save`
