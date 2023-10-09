# Installation

**NB!** The tool relies on [faiss](https://github.com/facebookresearch/faiss) for fast similarity search. To install faiss, please follow the [official installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

=== "Latest (w/o faiss)"
    ``` sh
    pip install hadal
    ```

=== "Development"
    ``` sh
    pip install hadal[dev]
    ```

=== "No dependencies"
    ``` sh
    pip install hadal --no-deps
    ```

=== "Ignore-installed"
    ``` sh
    pip install hadal --ignore-installed LIBTOIGNORE1 LIBTOIGNORE2
    ```
