# LogPred


##  results

|       |            | HDFS |     |
| :----:|:----:|:----:|:----:|
| **Model** | **Precision** | **Recall** | **F1** |
| DeepLog | 0.958 | 0.933 | 0.945 |
| AE | 0.713 | 0.488 | 0.579 |
| LogAnomaly | 0.97 | 0.94 | 0.96 |

|       |            | BG/L |     |
| :----:|:----:|:----:|:----:|
| **Model** | **Precision** | **Recall** | **F1** |
| DeepLog | 0.516 | 0.997 | 0.681 |
| AE | 0.538 |0.848 |0.658 |
| LogAnnomaly  | 0.96 | 0.94 | 0.95 |

**For DeepLog and its attention variants**
```
python train.py -model dl -num_layers 2 -hidden_size 128 -window_size 10 -dataset hd -epoch 300
python predict.py -model dl -num_layers 2 -hidden_size 128 -window_size 10 -dataset hd -epoch 300 -num_candidates 9
```
**For auto-encoder/VAE**
```
python train_ae.py -model vae -num_layers 2 -hidden_size 128 -window_size 10 -dataset hd -epoch 300 -dropout 0.1
python predict_ae.py -model vae -num_layers 2 -hidden_size 128 -window_size 10 -dataset hd -epoch 300 -dropout 0.1 -error_threshold 1.5
```

**For k-means clustering on trained latent space**
```
python clustering_fit.py -model vae -num_layers 2 -hidden_size 128 -window_size 10 -dataset hd -epoch 300 -dropout 0.1 -k 5 -iter 30
python clustering_pred.py -model vae -num_layers 2 -hidden_size 128 -window_size 10 -dataset hd -epoch 300 -dropout 0.1 -k 5 -threshold 3
```


BG/L dataset can be downloaded at [here](https://zenodo.org/record/3227177)

Then use [Drain](https://github.com/logpai/logparser/blob/master/demo/Drain_demo.py) to parse it but replace the original one with LogPred/bgl/Drain_demo.py

## Requirements

* Pytorch version 1.3

* Python 3

Note that the parser Drain requires python 2.7


## References
[1] : [DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf)

[2] : [Thanks for wuyifan18's PyTorch Implementation of DeepLog](https://github.com/wuyifan18/DeepLog)

[3] : [Thanks for Drain Paser Inplementation](https://github.com/logpai/logparser)

[4] : [Thanks for VAE Implementation](https://github.com/tejaslodaya/timeseries-clustering-vae)

[5] : [LogAnomaly](https://www.ijcai.org/Proceedings/2019/0658.pdf)
