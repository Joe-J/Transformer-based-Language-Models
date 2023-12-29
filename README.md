# Transformer-based Language Models
Implement two language modeling tasks based on vanilla Transformers. The datasets are PTB (Penn Treebank Dataset ) and Wikitext-2.
## first task: different tokenizer
The first task is based on word-level inputs and the second is based on byte pair encoding.
Make comparisons on the two types of inputs and list the results in a table (or in a figure).
### work-level inputs 
Split sentence into word.
```python
train_iter = PennTreebank(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = PennTreebank()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)
```
### BPE inputs

BPE is a popular sub-word tokenizer  which used by GPT2 and many other LLM.
```python
model_name = "gpt2"
tokenizer_bpe = GPT2Tokenizer.from_pretrained(model_name)
train_iter, val_iter, test_iter = PennTreebank()
def text_generator(raw_text_iter):
    for item in raw_text_iter:
        yield tokenizer_bpe.tokenize(item)
vocab = build_vocab_from_iterator(text_generator(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer_bpe.tokenize(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
```
### Comparisons
 I only train 3 epoch,the results show that BPE tokenizer is much butter than word-level tokenizer.
| Dataset      | Tokenization | Test Loss | Test Perplexity |
|--------------|--------------|-----------|-----------------|
| WiKi-Text2   | Word-level         | 5.50      | 245.35          |
| WiKi-Text2   | BPE          | 4.65      | 104.81          |
| PTB          | BPE          | 4.53      | 93.14           |
| PTB          | Word-level         | 5.23      | 187.41          |
    
## Second task: different attention module
Replace the standard self-attention module with linearized attention (Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (arxiv.org)) and make comparisons.

### Model architecture
I use TransformerEncoder,followed by [the tutorial of pytorch](https://pytorch.org/tutorials/beginner/transformer_tutorial.html). The more details shows on jupyter.

Here is the code of linear attention.
```python
def linear_attn(q, k, v, kv_mask = None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    q = q * dim ** -0.5

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)
```





~~The training stage is not stable，i only train 10 epoch,so there is no big difference.~~
**Mistakes:** I design a function to run training process, but i reference model for several times which leads to program conflicts,so the loss is too high, now i fix this problem, here is the new results.

|  Dataset | Attention Type    | Tokenization | Test Loss | Test Perplexity |
---------|-------------------|--------------|-----------|-----------------|
| PTB              | Self Attention    | word          | 5.35      | 210.35           |
| PTB              |Linear Attention  | word          | 5.45      | 232.69        |

## Citations
```
@techreport{zhuiyiroformer,
    title   = {linear-attention-transformer},
    author  = {Phil Wang},
    year    = {2021},
    url     = "https://github.com/lucidrains/linear-attention-transformer"
}
```
