# hadal

**hadal** ```/ˈheɪdəl/``` is a tool for parallel sentence mining with pretrained models.

🚧🚧🚧The project is under active development. The changes in the API can be breaking. 🚧🚧🚧


## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.


## Implemented methods

| Method                                                                    | Alignment type |
| ------------------------------------------------------------------------- | -------------- |
| **[margin-based](hadal/parallel_sentence_mining/margin_based/README.md)** | one-to-one     |
| soon...                                                                   |                |


## Quickstart

See demo file [demo.py](demo.py) for more details.

```python
$ python demo.py
```

```python
# score, source_sentence, target_sentence
[
    (1.5549, "I think I like wine now.", "Je pense que j'aime le vin maintenant."),
    (1.5079, "She eats one apple every day.", "Elle mange une pomme chaque jour."),
    (1.4353, "They serve pizza dogs in the cafeteria.", "Ils vendent des hot-dogs à la cafétéria."),
    (0.4112, "Empty sentence.", "Ce jeu se joue sur le vaisseau spatial."),
]
```
