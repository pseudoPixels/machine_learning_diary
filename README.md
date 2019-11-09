# Machine Learning Playground
[![Sponsors on Open Collective](https://img.shields.io/badge/ML-Tensorflow-orange.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/ML-scikit--learn-blue.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/ML-Keras-lightgrey.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/python-3.6-brightgreen.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](#sponsors)
[![Sponsors on Open Collective](https://img.shields.io/badge/debian-10+-blue.svg?style=flat-square)](#sponsors)




## Table of Contents

### 1.0 Tools or Concepts
#### 1.1 LSTM Autoencoders
* [LSTM Autoencoders with Keras](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/LSTM_Autoencoders_with_Keras/LSTM_Autoencoders_with_Keras.ipynb) 
 :: An Autoencoder is a type of Artificial Neural Network model that learns a 
 compressed representation of input (i.e., [Wiki](https://en.wikipedia.org/wiki/Autoencoder)). 
 LSTM autoencoder is an implementation for compressed sequence representation 
 for Encoder-Decoder LSTM. The Encoder part can compress the original input sequence 
 to a fixed length, which can be used as feature vector for other supervised learning 
 algorithms or even data visualization.

#### 1.2 TensorFlow
* [Linear Model Using TensorFlow]

#### 1.3 PyTorch
* [Name Classification using PyTorch]
 
 
### 2.0 Sequence To Sequence Learning & Language Models
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
  
   
### 3.0 Time Series Prediction
* [Time Series Forecasting with LSTM Autoencoders](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/time_series_forecasting_using_LSTM_Autoencoder/time_series_forecasting_using_LSTM_Autoencoder.ipynb)
 :: Time Series Forecasting on the `Shampoo Sale Dataset`. The dataset contains information of monthy shampoo sales for 36 consecutive months, i.e., time series. We train on the first 24 months using LSTM. We then evaluate the model for shampoo sales forecasting on the last 12 test months of the dataset.


### 4.0 Kaggle Competitions
* [Titanic Passenger Survival Prediction](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/Titanic/Titanic.ipynb) :: In this challenge, we need to complete the analysis of what sorts of people were likely to survive. Applying the tools of machine learning to predict which passengers survived the tragedy. The challenge
description can be found [here in Kaggle](https://www.kaggle.com/c/titanic/overview)

* [House Price Prediction](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/Kaggle_HousePricePrediction/KaggleHousePricePrediction.ipynb) :: Developing predictive machine learning model for predicting
house prices. The detailed description of the machine learning problem can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)



### 5.0 Extra
* [Visualizations]()
    * [Seaborn Visualization](https://github.com/pseudoPixels/machine_learning_diary/blob/master/notebooks/Visualizations/SeabornVizualization.ipynb) :: Some useful and frequently used Seaborn Visualizations.
* [Maths]()
    * [Differential Calculas]()


## Installation
Use the following commands to install the conda virtual environment for installing all the requirements and notebooks:
```buildoutcfg
git clone https://github.com/pseudoPixels/machine_learning_diary.git
cd machine_learning_diary
conda create -n python3.6 python=3.6
conda activate python3.6
pip install -r requirements.txt
jupyter notebook
```