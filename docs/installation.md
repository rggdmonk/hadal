# Installation

**NB!**

* The tool relies on [**faiss**](https://github.com/facebookresearch/faiss) for fast similarity search. To install faiss, please follow the [official installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).
* [**pytorch**](https://pytorch.org/get-started/locally/) is also required.

=== "Latest (w/o faiss and pytorch)"
    ``` sh
    pip install hadal
    ```

=== "Development"
    ``` sh
    pip install hadal[dev]
    pip install hadal[docs]
    ```

=== "No dependencies"
    ``` sh
    pip install hadal --no-deps
    ```
