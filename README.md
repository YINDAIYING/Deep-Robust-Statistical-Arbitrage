# Deep-Robust-Statistical-Arbitrage
This repository stores the implementation of the paper "DETECTING DATA-DRIVEN ROBUST STATISTICAL ARBITRAGE STRATEGIES WITH DEEP NEURAL NETWORKS". 

# Abstract
We present an approach, based on deep neural networks, that allows identifying robust statistical arbitrage strategies in financial markets. Robust statistical arbitrage strategies refer to self-financing trading strategies that enable profitable trading under model ambiguity. The presented novel methodology does not suffer from the curse of dimensionality nor does it depend on the identification of cointegrated pairs of assets and is therefore applicable even on high-dimensional financial markets or in markets where classical pairs trading approaches fail. Moreover, we provide a method to build an ambiguity set of admissible probability measures that can be derived from observed market data. Thus, the approach can be considered as being model-free and entirely data-driven. We showcase the applicability of our method by providing empirical investigations with highly profitable trading performances even in 50 dimensions, during financial crises, and when the cointegration relationship between asset pairs stops to persist

# Implementation
Our implementation is based on the deep learning engine Pytorch.
To execute the code, one simply place the datasets(start and end dates aligned) in the same directory as the .py file and adjust the training/testing date. (CUDA is required)
There is only one implementation for high dimensional assets, which is applied to the cases of 10-50 assets in the paper. To understand the code better, please refer to the high_dimension code, where you can find more detailed comments.

# Data
API has been provided for downloading data from Yahoo Finance. This includes the data for the pairs trading example (XOM and BP) and all the high dimensional examples.
However, the data for STOXX50E and S&P500 are not provided for legal reasons. 

Enjoy! :)
