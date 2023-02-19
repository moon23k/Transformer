# Transformer_Anchors

> The main purpose of this repo is to implement **Transformer Model** in three NLG tasks from scratch and measure its performance. 
Each task is Neural Machine Translation, Dialogue Generation, Abstractive Text Summarization. The model architecture has implemented by referring to the famous **Attention is All you Need** paper, and WMT14, Daily-Dialogue, Daily-CNN datasets have used for each task.
Machine translation and Dialogue generation deal with relatively short sequences, but summarization task covers long sequences. Since it is difficult to properly handle long sentences with only the basic Encoder-Decoder structure, hierarchical encoder structure is used for summary task.
Except for that, all configurations are the same for the three tasks.

<br>
<br>

## Model desc
> Natural Language Sequence is inevitably a time-series data. In order to consider the time series aspect, the RNN structure had been considered as the only option. But Transformer broke this conventional prejudice and showed remarkable achievements by only using Attention Mechanism without any RNN Layer. Existing RNN models always had two chronic problems. First is a vanishing gradient problem which is apparent as the sequence length gets longer. Second is Recurrent Operation process itself, which makes parallel processing difficult. But the Transformer solved these problems only with Attentions. As a result, the architecture not only performs well in a variety of NLP tasks, but is also fast in speed.


<br>
<br>

## Configurations
| &emsp; **Vocab Config**                            | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                               | :---                                    | :---                                     |
| **`Vocab Size:`** &hairsp; `30,000`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Vocab Type:`** &hairsp; `BPE`                   | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`        | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

<br>
<br>


## Results
> **Training Results**

<center>
  <img src="https://user-images.githubusercontent.com/71929682/201269096-2cc00b2f-4e8d-4071-945c-f5a3bfbca985.png" width="90%" height="70%">
</center>


</br>

> **Test Results**

</br>
</br>


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
python3 run.py -task [nmt, dialog, sum] -mode [train, test, inference] -search [greedy, beam]
```


<br>
<br>

## Reference
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
