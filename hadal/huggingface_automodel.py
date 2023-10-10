"""This module contains the `class HuggingfaceAutoModel` that can be used to encode text using a Huggingface AutoModel."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy
import torch
from tqdm.autonotebook import trange
from transformers import AutoConfig, AutoModel, AutoTokenizer

from hadal.custom_logger import default_custom_logger

if TYPE_CHECKING:
    import pathlib


class HuggingfaceAutoModel:
    """Class to encode text using a Huggingface AutoModel.

    Methods:
        encode: Encode text using a Huggingface AutoModel.
    """

    def __init__(
        self,
        model_name_or_path: str | pathlib.Path,
        device: str | None = None,
        *,
        enable_logging: bool = True,
        log_level: int | None = logging.INFO,
    ) -> None:
        """Initialize HuggingfaceAutoModel object.

        Args:
            model_name_or_path (str | pathlib.Path): Name or path to the pre-trained model.
            device (str | None, optional): Device for the model.
            enable_logging (bool, optional): Logging option.
            log_level (int | None, optional): Logging level.
        """
        if enable_logging is True:
            self.logger = default_custom_logger(name=__name__, level=log_level)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Pytorch device: %s", device)

        self._target_device = torch.device(device)

        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

        if model_name_or_path is not None and model_name_or_path != "":
            self.logger.info("Load huggingface model: %s", model_name_or_path)

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        output_value: str = "pooler_output",
        convert_to: str | None = None,
        *,
        normalize_embeddings: bool = False,
        device: str | None = None,
    ) -> list[torch.Tensor] | torch.Tensor | numpy.ndarray:
        """Encode text using a Huggingface AutoModel.

        Args:
            sentences (str | list[str]): The sentences to encode.
            batch_size (int, optional): The batch size.
            output_value (str, optional): Model output type. Can be `pooler_output` or `last_hidden_state`.
            convert_to (str | None, optional): Convert the embeddings to `torch` or `numpy` format. If `torch`, it will return a `torch.Tensor`. If `numpy`, it will return a `numpy.ndarray`. If `None`, it will return a `list[torch.Tensor]`.
            normalize_embeddings (bool, optional): Normalize the embeddings.
            device (str | None, optional): Device for the model.

        Raises:
            NotImplementedError: If the `output_value` is not implemented.

        Returns:
            all_embeddings (list[torch.Tensor] | torch.Tensor | numpy.ndarray): The embeddings of the sentences.
        """
        if device is None:
            device = self._target_device

        self.model.to(device)
        self.logger.info("Encoding on pytorch device: %s", device)

        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        length_sorted_idx = numpy.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            inputs = self.tokenizer(sentences_batch, return_tensors="pt", truncation=True, padding=True)
            inputs = batch_to_device(batch=inputs, target_device=device)

            with torch.no_grad():
                outputs = self.model(**inputs)

                if output_value == "pooler_output":
                    embeddings = outputs.pooler_output
                elif output_value == "last_hidden_state":
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    msg = f"output_value=`{output_value}` not implemented"
                    raise NotImplementedError(msg)

                if normalize_embeddings is True:
                    # apply L2 normalization to the embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.extend(embeddings)

        all_embeddings: list[torch.Tensor] = [all_embeddings[idx] for idx in numpy.argsort(length_sorted_idx)]

        if convert_to == "torch":
            all_embeddings: torch.Tensor = torch.stack(all_embeddings)
        elif convert_to == "numpy":
            all_embeddings: numpy.ndarray = torch.stack(all_embeddings).numpy()

        return all_embeddings

    def _text_length(self, text: list[str] | list | str) -> int:
        """Calculate the length of the given sentences.

        Args:
            text (list[str] | list | str): The sentences.

        Raises:
            TypeError: Input cannot be a `dict`.
            TypeError: Input cannot be a `tuple`.

        Returns:
            length (int): The length of the text.
        """
        if isinstance(text, dict):
            msg = "Input cannot be a `dict`."
            raise TypeError(msg)
        if isinstance(text, tuple):
            msg = "Input cannot be a `tuple`."
            raise TypeError(msg)

        if not hasattr(text, "__len__"):  # no len() method
            return 1
        if len(text) == 0:  # empty string or list
            return len(text)
        return sum([len(t) for t in text])  # sum of length of individual strings


def batch_to_device(batch, target_device: torch.device):  # noqa: ANN201, ANN001
    """Move a batch of tensors to the specified device."""
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
