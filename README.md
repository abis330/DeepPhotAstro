# early_lsst_classification (Final novel model name: *DeepPhotAstro*)
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


Required linux utilities:
1. awk
2. sed
3. cut
4. gnuplot


Data:
1. Training:
   1. training_set.csv
   2. training_set_metadata.csv
2. Test:
   1. test_set.csv
   2. test_set_metadata.csv


Summary of Object Types:<br/>
<img src="https://github.com/abinashsinha330/early_lsst_classification/blob/master/data_summary.png" width=600 align='middle'>


Types of models being tested:
1. Plain RNN (using GRU-based RNN):<br/>
   <img src="https://github.com/abinashsinha330/early_lsst_classification/blob/master/plain_rnn/model.png" width=500 align='middle'>

2. LSTM
   1. Phased-LSTM
      1. Variant 1: Here the inputs are only flux values and flux error values for each of the passbands (total 6 flux values and 6 flux error values)
      2. Variant 2: Here the inputs are flux values, flux error values and source wavelengths (total 6 flux values, 6 flux error values, 6 source wavelengths where there is zero value when there is zero flux value for the same)
      3. Variant 3: Inputs are flux values without pass band distinction, flux error values without pass band distinction, passband indicator (1, 2, 3, 4, 5, 6), source wavelengths (Here the validation accuracy is below 50% for 50 epochs and is not stable for training.
   2. Time-LSTM
3. Self-Attention (Transformer's Encoder Architecture)
4. Classical ML


Folders:
- **/plain_rnn**
- **/lstm**
- **/self-attention**
- **/classical_ml**
- **/misc_experiments** contains other experiments understanding feasibility of an idea like active learning
- **/data**


Benchmark models being tested against:
1. **Avocado**
2. **RAPID**
