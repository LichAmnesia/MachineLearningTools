# 三层MLP的结果
```
... loading data
... building the model
... training
epoch 1, minibatch 2500/2500, validation error 9.620000 %
     epoch 1, minibatch 2500/2500, test error of best model 10.090000 %
epoch 2, minibatch 2500/2500, validation error 8.610000 %
     epoch 2, minibatch 2500/2500, test error of best model 8.740000 %
epoch 3, minibatch 2500/2500, validation error 8.000000 %
     epoch 3, minibatch 2500/2500, test error of best model 8.160000 %
epoch 4, minibatch 2500/2500, validation error 7.600000 %
     epoch 4, minibatch 2500/2500, test error of best model 7.790000 %
epoch 5, minibatch 2500/2500, validation error 7.300000 %
     epoch 5, minibatch 2500/2500, test error of best model 7.590000 %
epoch 6, minibatch 2500/2500, validation error 7.020000 %
     epoch 6, minibatch 2500/2500, test error of best model 7.200000 %
epoch 7, minibatch 2500/2500, validation error 6.680000 %
     epoch 7, minibatch 2500/2500, test error of best model 6.990000 %
epoch 8, minibatch 2500/2500, validation error 6.260000 %
     epoch 8, minibatch 2500/2500, test error of best model 6.730000 %
epoch 9, minibatch 2500/2500, validation error 5.970000 %
     epoch 9, minibatch 2500/2500, test error of best model 6.510000 %
epoch 10, minibatch 2500/2500, validation error 5.740000 %
     epoch 10, minibatch 2500/2500, test error of best model 6.190000 %
Optimization complete. Best validation score of 5.740000 % obtained at iteration 25000, with test performance 6.190000 %
The code for file MLP_comment.py ran for 4.05m
```

# 四层MLP的结果
```
➜  theano git:(master) ✗ python MLP_comment.py
... loading data
... building the model
('classifier.params = ', [W, b, W, b, W, b])
... training
epoch 1, minibatch 2500/2500, validation error 9.080000 %
     epoch 1, minibatch 2500/2500, test error of best model 9.510000 %
epoch 2, minibatch 2500/2500, validation error 7.990000 %
     epoch 2, minibatch 2500/2500, test error of best model 8.340000 %
epoch 3, minibatch 2500/2500, validation error 7.190000 %
     epoch 3, minibatch 2500/2500, test error of best model 7.670000 %
epoch 4, minibatch 2500/2500, validation error 6.680000 %
     epoch 4, minibatch 2500/2500, test error of best model 7.130000 %
epoch 5, minibatch 2500/2500, validation error 5.960000 %
     epoch 5, minibatch 2500/2500, test error of best model 6.540000 %
epoch 6, minibatch 2500/2500, validation error 5.390000 %
     epoch 6, minibatch 2500/2500, test error of best model 6.020000 %
epoch 7, minibatch 2500/2500, validation error 4.860000 %
     epoch 7, minibatch 2500/2500, test error of best model 5.590000 %
epoch 8, minibatch 2500/2500, validation error 4.530000 %
     epoch 8, minibatch 2500/2500, test error of best model 5.180000 %
epoch 9, minibatch 2500/2500, validation error 4.250000 %
     epoch 9, minibatch 2500/2500, test error of best model 4.840000 %
epoch 10, minibatch 2500/2500, validation error 3.980000 %
     epoch 10, minibatch 2500/2500, test error of best model 4.540000 %
Optimization complete. Best validation score of 3.980000 % obtained at iteration 25000, with test performance 4.540000 %
The code for file MLP_comment.py ran for 11.43m
```

