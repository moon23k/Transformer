## Transformer

> This repository provides code for implementing the Transformer from scratch and for comparing its performance with the Transformer model provided by PyTorch. The goal is to deepen understanding of the Transformer's code implementation. While keeping the model size and hyperparameters consistent, the focus is on identifying differences that arise from the code implementation approach. Furthermore, all experimental variables, including data and training parameters, have been standardized. Datasets for machine translation, conversation generation, and document summarization have been chosen to assess the natural language generation capabilities.

<br><br>

## Model desc

### ⚫ Scratch Model

> This is a model directly implemented in code, based on the Transformer introduced in the paper "Attention is All You Need." It has been built by implementing and declaring the functionality of every component from start to finish. While the implementation may be more intricate than models offered by well-known libraries, it provides the advantage of easy customization, allowing users to add or remove features according to their specific goals.

<br>

### ⚫ Torch Model

> This is a model implemented simply using the nn.TransformerEncoderLayer and nn.TransformerDecoderLayer modules provided by PyTorch. It offers the advantage of simplicity in implementation. Although, for code comprehension, the model was configured by separately declaring Encoder and Decoder, it is worth noting that a more straightforward model can also be constructed using only the nn.Transformer class.

<br><br>

## Setup
The default values for experimental variables are set as follows, and each value can be modified by editing the config.yaml file. <br>

| **Tokenizer Setup**                         | **Model Setup**                   | **Training Setup**                |
| :---                                        | :---                              | :---                              |
| **`Tokenizer Type:`** &hairsp; `BPE`        | **`Input Dimension:`** `15,000`   | **`Epochs:`** `10`                |
| **`Vocab Size:`** &hairsp; `15,000`         | **`Output Dimension:`** `15,000`  | **`Batch Size:`** `32`            |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` | **`Hidden Dimension:`** `256`     | **`Learning Rate:`** `5e-4`       |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]` | **`PFF Dimension:`** `512`        | **`iters_to_accumulate:`** `4`    |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]` | **`Num Layers:`** `3`             | **`Gradient Clip Max Norm:`** `1` |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]` | **`Num Heads:`** `8`              | **`Apply AMP:`** `True`           |

<br>To shorten the training speed, techiques below are used. <br> 
* **Accumulative Loss Update**, as shown in the table above, accumulative frequency has set 4. <br>
* **Application of AMP**, which enables to convert float32 type vector into float16 type vector.

<br><br>

## Result

### ⚫ Machine Translation
| Model | BLEU | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| Scratch | 13.61 | 0m 58s | 0.20GB | 1.12GB |
| Torch | 14.30 | 0m 43s | 0.20GB | 0.95GB |

<br>

### ⚫ Dialogue Generation
| Model | ROUGE | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| Scratch | 1.45 | 0m 58s | 0.20GB | 0.97GB |
| Torch | 2.86 | 0m 41s | 0.20GB | 0.85GB |

<br>

### ⚫ Text Summarization
| Model | ROUGE | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| Scratch | 6.10 | 6m 13s | 0.21GB | 5.85GB |
| Torch | 7.91 | 2m 41s | 0.21GB | 2.90GB |

<br><br>


## How to Use
```
├── ckpt                    --this dir saves model checkpoints and training logs
├── config.yaml             --this file is for setting up arguments for model, training, and tokenizer 
├── data                    --this dir is for saving Training, Validataion and Test Datasets
├── model                   --this dir contains files for Deep Learning Model
│   ├── common.py
│   ├── __init__.py
│   ├── scratch_model.py
│   └── torch_model.py
├── module                  --this dir contains a series of modules
│   ├── data.py
│   ├── generate.py
│   ├── __init__.py
│   ├── model.py
│   ├── test.py
│   └── train.py
├── README.md
├── run.py                 --this file includes codes for actual tasks such as training, testing, and inference to carry out the practical aspects of the work
└── setup.py               --this file contains a series of codes for preprocessing data, training a tokenizer, and saving the dataset
```

<br>

**First clone git repo in your local env**
```
git clone https://github.com/moon23k/Transformer
```

<br>

**Download and Process Dataset via setup.py**
```
bash setup.py -task [all, translation, dialogue, summarization]
```

<br>

**Execute the run file on your purpose (search is optional)**
```
python3 run.py -task [translation, dialogue, summarization] \
               -mode [train, test, inference] \
               -model [scratch, torch] \
               -search [greedy, beam]
```


<br>

## Reference
* [**Attention is all you need**](https://arxiv.org/abs/1706.03762)
* [**Pytorch Official Page**](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
<br>
