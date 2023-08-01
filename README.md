## Transformer_Anchors

> The main purpose of this repo is to implement **Transformer Model** in three NLG tasks from scratch and measure its performance. 
Each task is Neural Machine Translation, Dialogue Generation, Abstractive Text Summarization. The model architecture has implemented by referring to the famous **Attention is All you Need** paper, and WMT14, Daily-Dialogue, Daily-CNN datasets have used for each task.
Machine translation and Dialogue generation deal with relatively short sequences, but summarization task covers long sequences. Since it is difficult to properly handle long sentences with only the basic Encoder-Decoder structure, hierarchical encoder structure is used for summary task.
Except for that, all configurations are the same for the three tasks.

<br><br>

## Model desc

> Natural Language Sequence is inevitably a time-series data. In order to consider the time series aspect, the RNN structure had been considered as the only option. But Transformer broke this conventional prejudice and showed remarkable achievements by only using Attention Mechanism without any RNN Layer. Existing RNN models always had two chronic problems. First is a vanishing gradient problem which is apparent as the sequence length gets longer. Second is Recurrent Operation process itself, which makes parallel processing difficult. But the Transformer solved these problems only with Attentions. As a result, the architecture not only performs well in a variety of NLP tasks, but is also fast in speed.


<br><br>

## Configurations
The default values for experimental variables are set as follows, and each value can be modified by editing the config.yaml file. <br>

| **Vocab Configs**                           | **Model Configs**                            | **Training Configs**              |
| :---                                        | :---                                         | :---                              |
| **`Tokenizer Type:`** &hairsp; `WordPiece`  | **`Input & Output Dimension:`** `5,000`      | **`Epochs:`** `10`                |
| **`Vocab Size:`** &hairsp; `30,000`         | **`Embedding Dimension:`** `256`             | **`Batch Size:`** `32`            |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` | **`Hidden Dimension:`** `512`                | **`Learning Rate:`** `5e-4`       |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]` | **`Position-wise Forward Dimension:`** `512` | **`iters_to_accumulate:`** `4`    |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]` | **`Num Layers:`** `2`                        | **`Gradient Clip Max Norm:`** `1` |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]` | **`Num Heads:`** `8`                         | **`Apply AMP:`** `True`           |

<br>To shorten the training speed, techiques below are used. <br> 
* **Accumulative Loss Update**, as shown in the table above, accumulative frequency has set 4. <br>
* **Application of AMP**, which enables to convert float32 type vector into float16 type vector.

<br><br>

## Result
|       | &emsp; Greedy BLEU Score &emsp; | &emsp; Beam BLEU Score &emsp; |
| :---: | :---: | :---: |
| &emsp; **Custom Model** &emsp; | - | - |
| &emsp; **Torch Model** &emsp;  | - | - |
| &emsp; **Hybrid Model** &emsp; | - | - |

<br><br>

## How to Use
**First clone git repo in your local env**
```
git clone https://github.com/moon23k/LSTM_Anchors
```

<br>

**Download and Process Dataset via setup.py**
```
bash setup.py -task [all, nmt, dialog, sum]
```

<br>

**Execute the run file on your purpose (search is optional)**
```
python3 run.py -task [nmt, dialog, sum] \
               -mode [train, test, inference] \
               -model [custom, torch, hybrid] \
               -search [greedy, beam]
```


<br><br>

## Reference
* [**Attention is all you need**](https://arxiv.org/abs/1706.03762)

<br>
