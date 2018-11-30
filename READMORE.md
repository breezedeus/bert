

# Create Pretraining Data

## 命令行说明

```bash
#!/usr/bin/env bash

export BERT_BASE_DIR='/Users/king/Documents/Ein/语料/BERT/chinese_L-12_H-768_A-12'
TEST_DATA_DIR='./data'

# create pretraining data
python create_pretraining_data.py \
  --input_file=$TEST_DATA_DIR/pretraining_test.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```



参数说明如下：

* `input_file`: 以英文`,`分割的多个输入文件；其中每行是一个句子，不同document之间以一个空行分割。下面有具体事例；

* `outut_file`: 以英文`,`分割的多个输出文件；如果是多个输出文件，那所有样本会以轮询的方法均匀写到这些文件中；

* `do_lower_case`: 是否对英文字母做小写变换。

* `max_predictions_per_seq`: 每个样本中做替换时，最多替换多少个tokens；

* `masked_lm_prob`: 每个样本中做替换时，最多替换多少比例的tokens；这个值会与`max_predictions_per_seq`一起生效。

  ```python
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))
  ```

* `dupe_factor`: 在产生训练样本对时，每个文件被重复处理的次数。

* `max_seq_length`: 产生序列（两个句子加起来，再加上3个占位符：[CLS], [SEP], [SEP]）的最大长度。如果产生的序列超过了这个长度，则会做截断。截断每次减少一个词，从两个句子中比较长的那个句子的头或尾随机选一个词扔掉。

  ```python
  def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length.
    
    `tokens_a` is a list, so is `tokens_b`.
    """
    while True:
      total_length = len(tokens_a) + len(tokens_b)
      if total_length <= max_num_tokens:
        break
  
      trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
      assert len(trunc_tokens) >= 1
  
      # We want to sometimes truncate from the front and sometimes from the
      # back to add more randomness and avoid biases.
      if rng.random() < 0.5:
        del trunc_tokens[0]
      else:
        trunc_tokens.pop()
  ```

* `short_seq_prob`: Probability of creating sequences which are shorter than the maximum length. Default value: `0.1`.

  ```python
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:  # 以小概率，随机选取一个产生序列（两个句子加起来）的长度
      target_seq_length = rng.randint(2, max_num_tokens)
  ```



替换一个token时，以`0.8`的概率把它替换为`[MASK]`，以`0.1`的概率把它替换为自己（也即不替换），以`0.1`的概率把它随机替换为一个token。

```python
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
```



`pretraining_test.txt`里长这样：

```
你好,您好
你好,你家住哪

我家住在天安门，你呢？
我住在桥下呢。。
```



## 输出说明

每个输出的样本长这样：

```bash
INFO:tensorflow:*** Example ***
INFO:tensorflow:tokens: [CLS] 我 家 住 在 天 安 [MASK] ， 你 [MASK] ？ [SEP] 我 ##薑 在 桥 下 呢 。 。 [SEP]
INFO:tensorflow:input_ids: 101 2769 2157 857 1762 1921 2128 103 8024 872 103 8043 102 2769 19009 1762 3441 678 1450 511 511 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:masked_lm_positions: 7 10 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:masked_lm_ids: 7305 1450 857 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:masked_lm_weights: 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
INFO:tensorflow:next_sentence_labels: 0
```

具体说明下：

### `tokens`

原始是这两句：

```
我家住在天安门，你呢？
我住在桥下呢。。
```



做了MASK后，长下面这样：

`[CLS] 我 家 住 在 天 安 [MASK] ， 你 [MASK] ？ [SEP] 我 ##薑 在 桥 下 呢 。 。 [SEP]`

做了3个替换：`门`和`呢`替换成了`[MASK]`，而`住`替换成了`##薑`。



### `input_ids`

对应的输入token id列表。会padding到指定长度。



### `input_mask`

其中为`0`表示是padding，`1`表示是真有token。



### `segment_ids`

表明哪些tokens来自第一个句子，哪些token来自第二个句子。



### `masked_lm_positions`

被替换的tokens所在的位置。



### `masked_lm_ids`

被替换的tokens原始的token ids。



### `masked_lm_weights`

被替换的tokens对应的权重值。正常就是`1.0`。



### `next_sentence_labels`

这两个句子是否构成**NSP (Next Sentence Pair)**。



这里的取值跟通常做法不一样，需要注意：

> **如果两个句子是真的有前后关系，这个值是`0`。如果第二个句子是随机选取的，这个值是`1`。**





# Run Classifier

