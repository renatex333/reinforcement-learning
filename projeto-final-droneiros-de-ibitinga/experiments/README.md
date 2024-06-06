## How to run the experiments

1. To run the hypothesis 1 experiments:

Run: 
```
python train_ppo_cnn.py
```

Change the `dispersion_inc` parameter to `0.05` and run again to have the training for both the scenarios.

To test the trained algorithm, run:

```
python test_trained_search.py --checkpoint <path/to/trained_checkpoint> [--see]
```

2. To run the experiment to the hypothesis 2:

Run:

```
python train_descentralized_ppo_cnn.py
```

3. To run the experiments to the hypothesis 3:

```
python train_ppo_cnn_comm.py
```

and

```
python train_ppo_cnn_lstm.py
```

to run the tests, use the scripts `test_trained_cnn_lstm.py` and `test_trained_search_comm.py` with the same parameters as the ones in H1.

4. To reproduce the experiment to the hypothesis 4:

```
python train_ppo_mlp_cov.py
```

and to see the trained algorithm, use the script `test_trained_cov_mlp.py` with the same parameters as the one mentioned in H1.

5. Finally, to reproduce the hypothesis 5 tests

Change the parameter `person_amount` in the environment (in file `train_ppo_cnn.py`) to 4 and run:

```
python train_ppo_cnn.py
```

You can test in the same way as H1.