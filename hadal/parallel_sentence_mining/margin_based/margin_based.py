"""This module contains the `class MarginBasedPipeline` that implements the margin-based pipeline."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hadal.custom_logger import default_custom_logger
from hadal.faiss_search import FaissSearch
from hadal.huggingface_automodel import HuggingfaceAutoModel
from hadal.parallel_sentence_mining.margin_based.margin_based_tools import MarginBased

if TYPE_CHECKING:
    import pathlib

    import numpy


class MarginBasedPipeline:
    """Class that implements the margin-based pipeline.

    Methods:
        make_alignments: Make sentence alignments.
    """

    def __init__(
        self,
        model_name_or_path: str | pathlib.Path,
        model_device: str | None = None,
        faiss_device: str | None = None,
        *,
        enable_logging: bool = True,
        log_level: int | None = logging.INFO,
    ) -> None:
        """Initialize a MarginBasedPipeline object.

        Args:
            model_name_or_path (str | pathlib.Path): Name or path to the pre-trained model.
            model_device (str | None, optional): Device for the model.
            faiss_device (str | None, optional): Device for the Faiss search. If `None`, it will use GPU if available, otherwise CPU.
            enable_logging (bool, optional): Logging option.
            log_level (int | None, optional):  Logging level.
        """
        self.model = HuggingfaceAutoModel(
            model_name_or_path=model_name_or_path,
            device=model_device,
            enable_logging=enable_logging,
        )
        self.align_method = MarginBased()
        self.faiss_search = FaissSearch(device=faiss_device, enable_logging=enable_logging)
        self.model_device = model_device
        self.faiss_device = faiss_device

        if enable_logging is True:
            self.logger = default_custom_logger(name=__name__, level=log_level)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

    def make_alignments(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        batch_size: int = 32,
        output_value: str = "pooler_output",
        convert_to: str = "numpy",
        *,
        normalize_embeddings: bool = True,
        knn_neighbors: int = 4,
        knn_metric: str = "inner_product",
        margin: str = "ratio",
        strategy: str = "max_score",
    ) -> list[tuple[numpy.float64, str, str]]:
        """Make sentence alignments.

        Args:
            source_sentences (list[str]): Source sentences.
            target_sentences (list[str]): Target sentences.
            batch_size (int, optional): The batch size.
            output_value (str, optional): Model output type. Can be `pooler_output` or `last_hidden_state`.
            convert_to (str, optional): Convert the embeddings to `torch` or `numpy` format. If `torch`, it will return a `torch.Tensor`. If `numpy`, it will return a `numpy.ndarray`. If `None`, it will return a `list[torch.Tensor]`.
            normalize_embeddings (bool, optional): Normalize the embeddings.
            knn_neighbors (int, optional): The number of nearest neighbors.
            knn_metric (str, optional): The metric to use for k-nearest neighbor search. Can be `inner_product` or `l2`.
            margin (str, optional): The margin function to use. Valid options are `ratio` and `distance`.
            strategy (str, optional): The strategy to use for selecting the best candidates.

        Returns:
            bitext_list (list[tuple[numpy.float64, str, str]]): The `list[tuple[score, source_sentence, target_sentence]]` of the best sentence alignments.
        """
        self.logger.info("Encoding embeddings for source sentences...")
        source_embeddings = self.model.encode(
            sentences=source_sentences,
            batch_size=batch_size,
            output_value=output_value,
            convert_to=convert_to,
            normalize_embeddings=normalize_embeddings,
        )
        self.logger.info("Encoding embeddings for target sentences...")
        target_embeddings = self.model.encode(
            sentences=target_sentences,
            batch_size=batch_size,
            output_value=output_value,
            convert_to=convert_to,
            normalize_embeddings=normalize_embeddings,
        )

        self.logger.info("Perform kNN in both directions...")
        self.logger.info("Perform kNN in source -> target direction")
        x2y_sim, x2y_ind = self.faiss_search.k_nearest_neighbors(
            source_embeddings,
            target_embeddings,
            k=knn_neighbors,
            knn_metric=knn_metric,
        )
        x2y_mean = x2y_sim.mean(axis=1)

        self.logger.info("Perform kNN in target -> source direction")
        y2x_sim, y2x_ind = self.faiss_search.k_nearest_neighbors(
            target_embeddings,
            source_embeddings,
            k=knn_neighbors,
            knn_metric=knn_metric,
        )
        y2x_mean = y2x_sim.mean(axis=1)

        self.logger.info("%s margin is selected", margin)
        chosen_margin = self.align_method.select_margin(margin=margin)

        self.logger.info("Compute forward and backward scores...")
        fwd_scores = self.align_method.margin_based_score_candidates(
            source_embeddings,
            target_embeddings,
            x2y_ind,
            x2y_mean,
            y2x_mean,
            margin=chosen_margin,
        )
        bwd_scores = self.align_method.margin_based_score_candidates(
            target_embeddings,
            source_embeddings,
            y2x_ind,
            y2x_mean,
            x2y_mean,
            margin=chosen_margin,
        )

        self.logger.info("Selecting best candidates...")
        indices, scores = self.align_method.select_best_candidates(
            source_embeddings,
            x2y_ind,
            fwd_scores,
            target_embeddings,
            y2x_ind,
            bwd_scores,
            strategy=strategy,
        )

        bitext_list = self.align_method.get_sentence_pairs(indices, scores, source_sentences, target_sentences)

        self.logger.info("Output sentences: pairs %d", len(bitext_list))

        return bitext_list
