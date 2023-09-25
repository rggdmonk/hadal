"""A wrapper class for Hugging Face's AutoModel that provides additional functionality for encoding text."""
import logging

import numpy
import torch
from tqdm.autonotebook import trange
from transformers import AutoConfig, AutoModel, AutoTokenizer

from hadal.cutstom_logger import default_custom_logger


class HuggingfaceAutoModel:
    """A wrapper class for Hugging Face's AutoModel that provides additional functionality for encoding text."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        *,
        enable_logging: bool = True,
        log_level: int | None = None,
    ) -> None:
        """Initializes the HuggingfaceAutoModel.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model to use.
            device (str | None, optional): The device to run the model on. Defaults to None.
            enable_logging (bool, optional): Whether to enable logging. Defaults to True.
            log_level (int | None, optional): The level of logging to use. Defaults to None.
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
        """Encodes the given sentences into embeddings.

        Args:
            sentences (str | list[str]): The sentences to encode.
            batch_size (int, optional): The batch size to use. Defaults to 32.
            output_value (str, optional): The type of output to use. Defaults to "pooler_output".
            convert_to (str | None, optional): The type to convert the output to. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize the embeddings. Defaults to False.
            device (str | None, optional): The device to run the model on. Defaults to None.

        Raises:
            NotImplementedError: If the output_value is not implemented.

        Returns:
            list[torch.Tensor] | torch.Tensor | numpy.ndarray: The embeddings of the sentences.
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
        """Calculates the length of the given text.

        Args:
            text (list[str] | list | str): The text to calculate the length of.

        Returns:
            int: The length of the text.
        """
        if isinstance(text, dict):
            msg = "Input cannot be a dictionary."
            return ValueError(msg)
        if isinstance(text, tuple):
            return ValueError("Input cannot be a tuple.")

        if not hasattr(text, "__len__"):  # no len() method
            return 1
        if len(text) == 0:  # empty string or list
            return len(text)
        return sum([len(t) for t in text])  # sum of length of individual strings


def batch_to_device(batch, target_device: torch.device):  # noqa: ANN201, ANN001
    """Move a batch of tensors to the specified device.

    Args:
        batch (Dict[str, torch.Tensor]): The batch of tensors to move.
        target_device (torch.device): The target device to move the tensors to.

    Returns:
        Dict[str, torch.Tensor]: The batch of tensors moved to the target device.
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
