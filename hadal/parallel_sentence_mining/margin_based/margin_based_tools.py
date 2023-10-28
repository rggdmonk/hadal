"""The module contains the `class MarginBased` that implements the margin-based scoring for parallel sentence mining."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy


class MarginBased:
    """Class that implements the margin-based scoring for parallel sentence mining.

    Methods:
        select_margin: Select the margin function.
        margin_based_score: Compute the margin-based score.
        margin_based_score_candidates: Compute the margin-based scores for a batch of sentence pairs.
        select_best_candidates: Select the best sentence pairs.
        get_sentence_pairs: Get the sentence pairs.
    """

    def __init__(self) -> None:
        """Initialize a MarginBased object."""

    def select_margin(self, margin: str = "ratio") -> Callable:
        """Select the margin function.

        Source: https://arxiv.org/pdf/1811.01136.pdf 3.1 Margin-based scoring

        Args:
            margin (str, optional): The margin function to use. Valid options are `ratio` and `distance`.

        Raises:
            NotImplementedError: If the given `margin` is not implemented.

        Returns:
            margin_func (Callable): The margin function.
        """
        if margin == "ratio":
            margin_func = lambda a, b: a / b  # noqa
        elif margin == "distance":
            margin_func = lambda a, b: a - b  # noqa
        else:
            msg = f"margin=`{margin}` is not implemented"
            raise NotImplementedError(msg)
        return margin_func

    def margin_based_score(
        self,
        source_embeddings: numpy.ndarray,
        target_embeddings: numpy.ndarray,
        fwd_mean: numpy.ndarray,
        bwd_mean: numpy.ndarray,
        margin_func: Callable,
    ) -> numpy.ndarray:
        """Compute the margin-based score.

        Source: https://arxiv.org/pdf/1811.01136.pdf 3.1 Margin-based scoring

        Args:
            source_embeddings (numpy.ndarray): Source embeddings.
            target_embeddings (numpy.ndarray): Target embeddings.
            fwd_mean (numpy.ndarray): The forward mean.
            bwd_mean (numpy.ndarray): The backward mean.
            margin_func (Callable): The margin function.

        Returns:
            score (numpy.ndarray): Margin-based score.
        """
        score = margin_func(source_embeddings.dot(target_embeddings), (fwd_mean + bwd_mean) / 2)

        return score

    def margin_based_score_candidates(
        self,
        source_embeddings: numpy.ndarray,
        target_embeddings: numpy.ndarray,
        candidate_inds: numpy.ndarray,
        fwd_mean: numpy.ndarray,
        bwd_mean: numpy.ndarray,
        margin: Callable,
    ) -> numpy.ndarray:
        """Compute the margin-based scores for a batch of sentence pairs.

        Args:
            source_embeddings (numpy.ndarray): Source embeddings.
            target_embeddings (numpy.ndarray): Target embeddings.
            candidate_inds (numpy.ndarray): The indices of the candidate target embeddings for each source embedding.
            fwd_mean (numpy.ndarray): The forward mean.
            bwd_mean (numpy.ndarray): The backward mean.
            margin (Callable): The margin function.

        Returns:
            scores (numpy.ndarray): The margin-based scores for the candidate pairs.
        """
        scores = numpy.zeros(candidate_inds.shape)
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                k = candidate_inds[i, j]
                scores[i, j] = self.margin_based_score(
                    source_embeddings[i],
                    target_embeddings[k],
                    fwd_mean[i],
                    bwd_mean[k],
                    margin,
                )
        return scores

    def select_best_candidates(
        self,
        source_embeddings: numpy.ndarray,
        x2y_ind: numpy.ndarray,
        fwd_scores: numpy.ndarray,
        target_embeddings: numpy.ndarray,
        y2x_ind: numpy.ndarray,
        bwd_scores: numpy.ndarray,
        strategy: str = "max_score",
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Select the best sentence pairs.

        Source: https://arxiv.org/pdf/1811.01136.pdf 3.2 Candidate generation and filtering (only max. score)

        Args:
            source_embeddings (numpy.ndarray): Source embeddings.
            x2y_ind (numpy.ndarray): Indices of the target sentences corresponding to each source sentence.
            fwd_scores (numpy.ndarray): Scores of the forward alignment between source and target sentences.
            target_embeddings (numpy.ndarray): Target embeddings.
            y2x_ind (numpy.ndarray): Indices of the source sentences corresponding to each target sentence.
            bwd_scores (numpy.ndarray): Scores of the backward alignment between target and source sentences.
            strategy (str, optional): The strategy to use for selecting the best candidates.

        Raises:
            NotImplementedError: If the given `strategy` is not implemented.

        Returns:
            - indices (numpy.ndarray): An array of indices representing the sentence pairs.
            - scores (numpy.ndarray): An array of scores representing the similarity between the sentence pairs.
        """
        if strategy == "max_score":
            fwd_best = x2y_ind[numpy.arange(source_embeddings.shape[0]), fwd_scores.argmax(axis=1)]
            bwd_best = y2x_ind[numpy.arange(target_embeddings.shape[0]), bwd_scores.argmax(axis=1)]

            indices = numpy.stack(
                [
                    numpy.concatenate([numpy.arange(source_embeddings.shape[0]), bwd_best]),
                    numpy.concatenate([fwd_best, numpy.arange(target_embeddings.shape[0])]),
                ],
                axis=1,
            )
            scores = numpy.concatenate([fwd_scores.max(axis=1), bwd_scores.max(axis=1)])

        else:
            msg = f"`{strategy}` is not implemented"
            raise NotImplementedError(msg)

        return indices, scores

    def get_sentence_pairs(
        self,
        indices: numpy.ndarray,
        scores: numpy.ndarray,
        source_sentences: list[str],
        target_sentences: list[str],
    ) -> list[tuple[numpy.float64, str, str]]:
        """Get the sentence pairs.

        Args:
            indices (numpy.ndarray): An array of indices representing the sentence pairs.
            scores (numpy.ndarray): An array of scores representing the similarity between the sentence pairs.
            source_sentences (list[str]): Source sentences.
            target_sentences (list[str]): Target sentences.

        Returns:
            bitext_list (list[tuple[numpy.float64, str, str]]): A list of tuples with score, source sentences and target sentences.
        """
        seen_src, seen_trg = set(), set()

        bitext_list = []

        for i in numpy.argsort(-scores):
            src_ind, trg_ind = indices[i]
            src_ind = int(src_ind)
            trg_ind = int(trg_ind)

            if src_ind not in seen_src and trg_ind not in seen_trg:
                seen_src.add(src_ind)
                seen_trg.add(trg_ind)
                rounded_score = numpy.round(scores[i], 4)
                bitext_list.append((rounded_score, source_sentences[src_ind], target_sentences[trg_ind]))

        return bitext_list
