# Deep-Robust-Statistical-Arbitrage
This repository stores the implementation of the paper "DETECTING DATA-DRIVEN ROBUST STATISTICAL ARBITRAGE STRATEGIES WITH DEEP NEURAL NETWORKS". 

# Abstract
We present an approach, based on deep neural networks, that allows identifying robust statistical arbitrage strategies in financial markets. Robust statistical arbitrage strategies refer
to trading strategies that enable profitable trading under model ambiguity. The presented novel
methodology allows to consider a large amount of underlying securities simultaneously and does not
depend on the identification of cointegrated pairs of assets, hence it is applicable on high-dimensional
financial markets or in markets where classical pairs trading approaches fail. Moreover, we provide
a method to build an ambiguity set of admissible probability measures that can be derived from
observed market data. Thus, the approach can be considered as being model-free and entirely datadriven. We showcase the applicability of our method by providing empirical investigations with
highly profitable trading performances even in 50 dimensions, during financial crises, and when the
cointegration relationship between asset pairs stops to persist.

# Content
The file "linear programming approach.ipynb" covers the linear programming example.

The file "1_asset (comparison with LP approach).py" covers the neural network approach in one asset setting. This is done in comparison with the linear programming approach.

The file "2_assets (STOXX&SP500) bad_market.py " covers the neural network approach in an overall decline market example.

The file "2_assets (XOM&BP) pairs_trading.py" covers the neural network approach in a situation where the pairs trading strategy fails.

The file "per_share transaction cost high_dimension.py" covers the neural network approach in high dimensional setting (10-50 assets) with per_share transaction cost applied.

The file "proportional transaction cost high_dimension.py" covers the neural network approach in high dimensional setting (10-50 assets) with proportional transaction cost applied.

The file "Online Learning.py" covers the approach which train the model with streamed data, that is, to fine-tune the model with the most recent incoming data, known as online-learning in the literature.

# Implementation
Our implementation is based on the deep learning engine Pytorch. Note that CUDA is needed for execution!
Gurobi is needed to excute the linear programming code. The package could be easily installed in Google Colab as demonstrated in the .ipynb file.
To execute the code, one simply place the datasets(start and end dates aligned) in the same directory as the .py file and adjust the training/testing date. For most of the examples, API has been provided to download the data.
There is only one implementation for high dimensional assets, which is applied to the cases of 10-50 assets in the paper. To understand the code better, please refer to the high-dimension code, where you can find more detailed comments.

# Data
API has been provided for downloading data from Yahoo Finance. This includes the data for the pairs trading example (XOM and BP) and all the high dimensional examples.
However, the data for STOXX50E and S&P500 are not provided for legal reasons. 

Enjoy! :)
