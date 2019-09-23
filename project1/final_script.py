#!/usr/bin/env python3
import utils as ut

# gather  and process red wine data
x = ut.process_data(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    ";",
)
ut.data_worker(x)

# run LDA and LR for red wine data
ut.cross_val_LDA(x.X.to_numpy(), x.Y, 5)
ut.cross_val_LR(x.X_int, x.Y, 5)
