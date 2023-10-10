# Margin-based

## Quickstart

``` py
import hadal

# (1) Prepare source and target sentences

source_test = [
        "I think I like wine now.",
        "She eats one apple every day.",
        "They serve pizza dogs in the cafeteria.",
        "Empty sentence.",
    ]

target_test = [
        "Je pense que j'aime le vin maintenant.",
        "Elle mange une pomme chaque jour.",
        "Ils vendent des hot-dogs à la cafétéria.",
        "Ce jeu se joue sur le vaisseau spatial.",
        "Barry est plus tard déchargé du corps après sa dernière bataille.",
    ]

# (2) Load model

model_name = "setu4993/LaBSE"

# (3) Load alignment config

alignment_config = hadal.MarginBasedPipeline(
        model_name_or_path=model_name,
        model_device="cpu",
        faiss_device="cpu"
    )

# (4) Make alignments

result = alignment_config.make_alignments(
        source_sentences=source_test,
        target_sentences=target_test,
        knn_neighbors=2,
    )

for ind, (score, src, tgt) in enumerate(result, start=1):
    print(f"Pair: {ind}")
    print(f"Score: {score}\nSource: {src}\nTarget: {tgt}\n")
```

```
# (5) Expected output

Pair: 1
Score: 1.5549
Source: I think I like wine now.
Target: Je pense que j'aime le vin maintenant.

Pair: 2
Score: 1.5079
Source: She eats one apple every day.
Target: Elle mange une pomme chaque jour.

Pair: 3
Score: 1.4353
Source: They serve pizza dogs in the cafeteria.
Target: Ils vendent des hot-dogs à la cafétéria.

Pair: 4
Score: 0.4112
Source: Empty sentence.
Target: Ce jeu se joue sur le vaisseau spatial.

```



## High-level description
1. Encode `source` and `target` sentences with the model ([LaBSE](https://arxiv.org/abs/2007.01852) is recommended, or you can try others from [huggingface](https://huggingface.co/models))
2. Find the k nearest neighbor sentences for all sentences in both directions. Choose a value of `k` between 4 and 16.
3. Score all possible sentence combinations using the formula mentioned in [Section 4.3](https://arxiv.org/pdf/1811.01136.pdf).
4. The pairs with the highest scores are most likely translated sentences. Note that the score can be larger than 1. You may need to set a threshold and ignore pairs below that threshold. A threshold of about 1.2 - 1.3 works well for high-quality results.


## Acknowledgements
* Original paper: [Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings](https://arxiv.org/pdf/1811.01136.pdf)
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) [implementation](https://github.com/UKPLab/sentence-transformers/blob/c5f93f70eca933c78695c5bc686ceda59651ae3b/examples/applications/parallel-sentence-mining/README.md).
