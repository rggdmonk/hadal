# What is Parallel Sentence Mining?

Parallel sentence mining is a process of searching parallel (translated) sentence pairs in monolingual corpora.

```title="source (English)" linenums="1"
This wine bar - restaurant promises you beautiful culinary surprises.
Every cup of coffee should create a personal moment of pleasure.
Some text that is not translated.
```

```title="target (French)" linenums="1"
Chaque tasse de café devrait créer un moment de plaisir personnel.
Deux
Ce bar à vins - restaurant vous promet de belles surprises culinaires.
```

The goal is to identify all translation pairs between the `source` and `target` sets of sentences.


| source                                                                | target                                                                 | index |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----- |
| This wine bar - restaurant promises you beautiful culinary surprises. | Ce bar à vins - restaurant vous promet de belles surprises culinaires. | 1 - 3 |
| Every cup of coffee should create a personal moment of pleasure.      | Chaque tasse de café devrait créer un moment de plaisir personnel.     | 2 - 1 |
