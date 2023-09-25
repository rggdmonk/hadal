"""This module contains the MarginBasedPipeline class, which is a pipeline for aligning parallel sentences using margin-based scoring."""
import logging

import numpy

from hadal.cutstom_logger import default_custom_logger
from hadal.faiss_search import FaissSearch
from hadal.huggingface_automodel import HuggingfaceAutoModel
from hadal.parallel_sentence_mining.margin_based.margin_based_tools import MarginBased


class MarginBasedPipeline:
    """Pipeline for aligning parallel sentences using margin-based scoring."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        model_device: str | None = None,
        faiss_device: str | None = None,
        *,
        enable_logging: bool = True,
        log_level: int | None = logging.INFO,
    ) -> None:
        """Initializes a MarginBasedPipeline object.

        Args:
            model_name_or_path (str | None, optional): The name or path of the Hugging Face model to use. Defaults to None.
            model_device (str | None, optional): The device to use for the model. Defaults to None.
            faiss_device (str | None, optional): The device to use for Faiss search. Defaults to None.
            enable_logging (bool, optional): Whether to enable logging. Defaults to True.
            log_level (int | None, optional): The logging level to use. Defaults to logging.INFO.
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
        """Compute sentence alignments between source and target sentences using margin-based scoring.

        Args:
            source_sentences: A list of strings representing the source sentences.
            target_sentences: A list of strings representing the target sentences.
            batch_size: An integer representing the batch size for encoding the sentences.
            output_value: A string representing the output value to use for encoding the sentences.
            convert_to: A string representing the format to convert the encoded embeddings to.
            normalize_embeddings: A boolean indicating whether to normalize the embeddings.
            knn_neighbors: An integer representing the number of nearest neighbors to consider in kNN search.
            knn_metric: A string representing the metric to use for kNN search.
            margin: A string representing the margin function to use for scoring.
            strategy: A string representing the strategy to use for selecting the best candidates.

        Returns:
            A list of tuples representing the aligned sentence pairs, where each tuple contains:
            - A float representing the alignment score.
            - A string representing the source sentence.
            - A string representing the target sentence.
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
