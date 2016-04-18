## Learning to Read Chest X-Rays
### Source code for the CVPR 2016 paper:
Learning to Read Chest X-Rays: Recurrent Neural Cascade Model for Automated Image Annotation ([arxiv](http://arxiv.org/abs/1603.08486))

#### The source code is based on the:
- [cifar.torch](https://github.com/szagoruyko/cifar.torch) by Sergey Zagoruyko, for the CNN training
- [char-rnn](https://github.com/karpathy/char-rnn) by Andrej Karpathy, for the RNN training

#### The overall pipeline and codes for preparing data, training, and sampling can be found in the [src/chestx](https://github.com/khcs/learning-to-read/tree/master/src/chestx)

#### Dataset can be downloaded from the Open-i chest x-ray subset ([link](https://openi.nlm.nih.gov/gridquery.php?q=&it=xg&sub=x))

#### Trained models can be downloaded:
- [CNN](https://drive.google.com/open?id=0B_g1jY2y9MKdeVpEb1FLVVA3Y28) - GoogLeNet model
- [RNN-LSTM](https://drive.google.com/open?id=0B_g1jY2y9MKdVGdXR0lyY1FpbVk), [RNN-GRU](https://drive.google.com/open?id=0B_g1jY2y9MKdR3pWSURYRkZWdjQ)

#### Required software packages:
- [Torch](http://torch.ch/)
- [Python 2.7](https://www.python.org/) with the following packages:
  - [Numpy](http://www.numpy.org/)
  - [Matplotlib](http://matplotlib.org/)
  - [Pandas](http://pandas.pydata.org/)
  - [Scikit-Learn](http://scikit-learn.org/stable/)
  - [Scikit-Image](http://scikit-image.org/)
  - [iPython Notebook](http://ipython.org/notebook.html) - Optional
  - [NLTK](http://www.nltk.org/) - Optional
