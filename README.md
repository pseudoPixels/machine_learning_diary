# Machine Learning Diary
[![Sponsors on Open Collective](https://img.shields.io/badge/ML-Tensorflow-orange.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/ML-scikit--learn-blue.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/ML-Keras-lightgrey.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/python-3.6-brightgreen.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/debian-10+-blue.svg?style=flat-square)](#sponsors)




## Table of Contents

* [Tools or Concepts](#tools)
  * [LSTM Autoencoders with Keras](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/LSTM_Autoencoders_with_Keras/LSTM_Autoencoders_with_Keras.ipynb) 
     :: An Autoencoder is a type of Artificial Neural Network model that learns a 
     compressed representation of input (i.e., [Wiki](https://en.wikipedia.org/wiki/Autoencoder)). 
     LSTM autoencoder is an implementation for compressed sequence representation 
     for Encoder-Decoder LSTM. The Encoder part can compress the original input sequence 
     to a fixed length, which can be used as feature vector for other supervised learning 
     algorithms or even data visualization.
* [Sequence To Sequence Learning & Language Models](#usage)
  * [Character Based Neural Language Model using LSTM RNN](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/character_based_neural_language_model/character_based_neural_language_model.ipynb)
     :: This is a character based neural language model. The LSTM model
     is trained using sequences of characters from a data source to predict
     the next possible character in the sequence. 
  * [Text Generative Model using LSTM RNN](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/character_based_neural_language_model/character_based_neural_language_model.ipynb)
     :: This is a word based neural language model. The LSTM model
     is trained using sequences of words from a data source to predict
     the next possible word in the sequence.
  * [Mathematical Addition using RNN](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/mathematical_addition_using_RNN/mathematical_addition_using_RNN.ipynb)  :: This is a sequence to sequence learning problem where the RNN model learns to add two numbers (as string). This is a simple example where, the input and output length are always same, unlike other language models or sequence problems.
  * [IMDB Sentiment Classification using LSTM]()  
* [Time Series Prediction](#usage)
  * [Time Series Forecasting with LSTM Autoencoders](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/time_series_forecasting_using_LSTM_Autoencoder/time_series_forecasting_using_LSTM_Autoencoder.ipynb)
     :: Time Series Forecasting on the `Shampoo Sale Dataset`. The dataset contains information of monthy shampoo sales for 36 consecutive months, i.e., time series. We train on the first 24 months using LSTM. We then evaluate the model for shampoo sales forecasting on the last 12 test months of the dataset.

* [Kaggle Competition](#usage)
    * [Titanic Passenger Survival Prediction](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/Titanic/Titanic.ipynb)

## Installation

```buildoutcfg
git clone https://github.com/pseudoPixels/machine_learning_diary.git
cd machine_learning_diary
conda create -n python3.6 python=3.6
conda activate python3.6
pip install -r requirements.txt
jupyter notebook
```