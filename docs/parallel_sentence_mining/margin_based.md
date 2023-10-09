# Margin-based

## Brief description

1. Encode `source` and `target` sentences with the model ([LaBSE](https://arxiv.org/abs/2007.01852) is recommended, or you can try others from [huggingface](https://huggingface.co/models))
2. Find the k nearest neighbor sentences for all sentences in both directions. Choose a value of `k` between 2 and 16.
3. Score all possible sentence combinations using the formula mentioned in [Section 4.3](https://arxiv.org/pdf/1811.01136.pdf).
4. The pairs with the highest scores are most likely translated sentences. Note that the score can be larger than 1. You may need to set a threshold and ignore pairs below that threshold. A threshold of about 1.2 - 1.3 works well for high-quality results.


## Acknowledgements
* Original paper: [Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings](https://arxiv.org/pdf/1811.01136.pdf)
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) [implementation](https://github.com/UKPLab/sentence-transformers/blob/c5f93f70eca933c78695c5bc686ceda59651ae3b/examples/applications/parallel-sentence-mining/README.md).
