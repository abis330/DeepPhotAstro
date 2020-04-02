# early_lsst_classification
Repository containing code for experiments conducted with regards to early time-series classification of light curves to identify type of each astronomical source responsible for a particular light curve emissions

Required Python Packages:
1. numpy
2. pandas
3. matplotlib
4. plotly
5. seaborn
6. scikit-learn
7. tensorflow
8. keras
9. pytorch

Types of models being tested:

1. Plain RNN (using GRU-based RNN)
2. LSTM
   a. Phased-LSTM
   b. Time-LSTM
3. Self-Attention (Transformer's Encoder Architecture)
4. Classical ML

Final novel model name: *DeepPhotTrans*

Benchmark models being tested against:
1. Avocado
2. RAPID

