"""This module provides a class for performing k-nearest neighbor search using the Faiss library."""
import logging

import faiss
import numpy

from hadal.cutstom_logger import default_custom_logger


class FaissSearch:
    """A class for performing k-nearest neighbor search using Faiss library."""

    def __init__(self, device: str | None = None, *, enable_logging: bool = True, log_level: int | None = logging.INFO) -> None:
        """Initialize FaissSearch object.

        Args:
            device (str | None, optional): The device to use for Faiss search. If None, it will use GPU if available, otherwise CPU. Defaults to None.
            enable_logging (bool, optional): Whether to enable logging. Defaults to True.
            log_level (int | None, optional): The logging level. Defaults to logging.INFO.
        """
        if enable_logging is True:
            self.logger = default_custom_logger(name=__name__, level=log_level)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

        if device is None:
            device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
            self.logger.info("Faiss device: %s", device)

        self._target_device = device

    def k_nearest_neighbors(
        self,
        source_embeddings: numpy.ndarray,
        target_embeddings: numpy.ndarray,
        k: int | None = 4,
        knn_metric: str = "inner_product",
        device: str | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Perform k-nearest neighbor search using Faiss.

        Args:
            source_embeddings (numpy.ndarray): The source embeddings to search from.
            target_embeddings (numpy.ndarray): The target embeddings to search in.
            k (int | None, optional): The number of nearest neighbors to return. Defaults to 4.
            knn_metric (str, optional): The metric to use for k-nearest neighbor search. Can be "inner_product" or "l2". Defaults to "inner_product".
            device (str | None, optional): The device to use for Faiss search. If None, it will use the device specified in the constructor. Defaults to None.

        Raises:
            NotImplementedError: If device is not "cpu".

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the distances and indices of the k-nearest neighbors.
        """
        if device is None:
            device = self._target_device

        self.logger.info("Perform k-nearest neighbor search...")

        if knn_metric == "inner_product":
            knn_metric = faiss.METRIC_INNER_PRODUCT
        elif knn_metric == "l2":
            knn_metric = faiss.METRIC_L2

        if device == "cpu":
            self.logger.info("Using faiss knn on CPU...")
            # https://github.com/facebookresearch/faiss/blob/d85601d972af2d64103769ab8d940db28aaae2a0/faiss/python/extra_wrappers.py#L330
            d, ind = faiss.knn(xq=source_embeddings, xb=target_embeddings, k=k, metric=knn_metric)
        else:
            exception_msg = "Faiss GPU is not implemented yet!"
            raise NotImplementedError(exception_msg)

        self.logger.info("Done k-nearest neighbor search!")
        return d, ind
