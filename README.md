# hadal

![PyPI - Version](https://img.shields.io/pypi/v/hadal)

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**hadal** ```/ˈheɪdəl/``` is a simple and efficient tool for mining and aligning sentences with pre-trained models.

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
