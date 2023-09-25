"""A module that implements MarginBased class for margin-based scoring for parallel sentence mining."""
from collections.abc import Callable

import numpy


class MarginBased:
    """A class that implements margin-based scoring for parallel sentence mining.

    Attributes:
        None

    Methods:
        select_margin: Selects the margin function based on the parameter.
        margin_based_score: Computes the margin-based score for a given source and target embedding.
        margin_based_score_candidates: Computes the margin-based score for a set of candidate pairs.
        select_best_candidates: Selects the best candidate pairs based on the strategy.
        get_sentence_pairs: Returns the sentence pairs with the highest margin-based scores.

    """

    def __init__(self) -> None:
        """Initializes a MarginBased object."""

    def select_margin(self, margin: str = "ratio") -> Callable:
        """Selects a margin function based on the given margin parameter.

        Args:
            margin (str, optional): The type of margin function to select. Valid options are "ratio" and "distance". Defaults to "ratio".

        Raises:
            NotImplementedError: If the given margin type is not implemented.

        Returns:
            Callable: The selected margin function.
        """
        # https://arxiv.org/pdf/1811.01136.pdf 3.1 Margin-based scoring

        if margin == "ratio":
            margin = lambda a, b: a / b  # noqa
        elif margin == "distance":
            margin = lambda a, b: a - b  # noqa
        else:
            msg = f"margin=`{margin}` is not implemented"
            raise NotImplementedError(msg)
        return margin

    def margin_based_score(
        self,
        source_embeddings: numpy.ndarray,
        target_embeddings: numpy.ndarray,
        fwd_mean: float,
        bwd_mean: float,
        margin: Callable,
    ) -> float:
        """Computes the margin-based score between source and target embeddings.

        Args:
            source_embeddings (numpy.ndarray): Embeddings of the source sentence.
            target_embeddings (numpy.ndarray): Embeddings of the target sentence.
            fwd_mean (float): Mean of forward scores.
            bwd_mean (float): Mean of backward scores.
            margin (Callable): Margin function to be used.

        Returns:
            float: Margin-based score between source and target embeddings.
        """
        # https://arxiv.org/pdf/1811.01136.pdf 3.1 Margin-based scoring

        return margin(source_embeddings.dot(target_embeddings), (fwd_mean + bwd_mean) / 2)

    def margin_based_score_candidates(
        self,
        source_embeddings: numpy.ndarray,
        target_embeddings: numpy.ndarray,
        candidate_inds: numpy.ndarray,
        fwd_mean: float,
        bwd_mean: float,
        margin: Callable,
    ) -> numpy.ndarray:
        """Computes the margin-based score for a set of candidate pairs of source and target embeddings.

        Args:
            source_embeddings (numpy.ndarray): The source embeddings.
            target_embeddings (numpy.ndarray): The target embeddings.
            candidate_inds (numpy.ndarray): The indices of the candidate target embeddings for each source embedding.
            fwd_mean (float): The forward mean.
            bwd_mean (float): The backward mean.
            margin (Callable): The margin function.

        Returns:
            numpy.ndarray: The margin-based scores for the candidate pairs.
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
        """Selects the best candidates for parallel sentences based on the given strategy.

        Args:
            source_embeddings (numpy.ndarray): Embeddings of the source sentences.
            x2y_ind (numpy.ndarray): Indices of the target sentences corresponding to each source sentence.
            fwd_scores (numpy.ndarray): Scores of the forward alignment between source and target sentences.
            target_embeddings (numpy.ndarray): Embeddings of the target sentences.
            y2x_ind (numpy.ndarray): Indices of the source sentences corresponding to each target sentence.
            bwd_scores (numpy.ndarray): Scores of the backward alignment between target and source sentences.
            strategy (str, optional): The strategy to use for selecting the best candidates. Defaults to "max_score".

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the indices and scores of the best candidates.
        """
        # https://arxiv.org/pdf/1811.01136.pdf 3.2 Candidate generation and filtering (only max. score)

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
        """Given a set of indices, scores, source sentences, and target sentences, returns a list of sentence pairs with their corresponding scores.

        The sentence pairs are selected based on the highest scores and the constraint that each source sentence and target sentence can only appear once in the list.

        Args:
            indices (numpy.ndarray): An array of indices representing the sentence pairs.
            scores (numpy.ndarray): An array of scores representing the similarity between the sentence pairs.
            source_sentences (list[str]): A list of source sentences.
            target_sentences (list[str]): A list of target sentences.

        Returns:
            list[tuple[numpy.float64, str, str]]: A list of sentence pairs with their corresponding scores.
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
