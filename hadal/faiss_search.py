"""This module contains the `class FaissSearch` that can be used to perform k-nearest neighbor search using the Faiss library."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import faiss

from hadal.custom_logger import default_custom_logger

if TYPE_CHECKING:
    import numpy


class FaissSearch:
    """Class to perform k-nearest neighbor search using the Faiss library.

    Methods:
        k_nearest_neighbors: Perform k-nearest neighbor search using Faiss.
    """

    def __init__(self, device: str | None = None, *, enable_logging: bool = True, log_level: int | None = logging.INFO) -> None:
        """Initialize FaissSearch object.

        Args:
            device (str | None, optional): Device for the Faiss search. If `None`, it will use GPU if available, otherwise CPU. Default is `None`.
            enable_logging (bool, optional): Logging option.
            log_level (int | None, optional): Logging level.
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
        k: int = 4,
        knn_metric: str = "inner_product",
        device: str | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Perform k-nearest neighbor search using Faiss.

        Args:
            source_embeddings (numpy.ndarray): The source embeddings.
            target_embeddings (numpy.ndarray): The target embeddings.
            k (int, optional): The number of nearest neighbors.
            knn_metric (str, optional): The metric to use for k-nearest neighbor search. Can be `inner_product` or `sqeuclidean`.
            device (str | None, optional): The device to use for Faiss search. If `None`, it will use GPU if available, otherwise CPU.

        Note:
            It is fully relying on the Faiss library for the k-nearest neighbor search `faiss.knn` and `faiss.gpu_knn`.

            - `inner_product` uses `faiss.METRIC_INNER_PRODUCT`
            - `sqeuclidean` uses `faiss.METRIC_L2` (squared Euclidean distance)

        Returns:
            - d (numpy.ndarray): The distances of the k-nearest neighbors.
            - ind (numpy.ndarray): The indices of the k-nearest neighbors.
        """
        if device is None:
            device = self._target_device

        self.logger.info("Perform k-nearest neighbor search...")

        if knn_metric == "inner_product":
            knn_metric = faiss.METRIC_INNER_PRODUCT
        elif knn_metric == "sqeuclidean":
            # squared Euclidean (L2) distance
            knn_metric = faiss.METRIC_L2

        if device == "cpu":
            self.logger.info("Using faiss knn on CPU...")
            # https://github.com/facebookresearch/faiss/blob/d85601d972af2d64103769ab8d940db28aaae2a0/faiss/python/extra_wrappers.py#L330
            d, ind = faiss.knn(xq=source_embeddings, xb=target_embeddings, k=k, metric=knn_metric)
        else:
            self.logger.info("Using faiss knn on GPU...")
            res = faiss.StandardGpuResources()
            d, ind = faiss.gpu_knn(res=res, xq=source_embeddings, xb=target_embeddings, k=k, metric_type=knn_metric)

        self.logger.info("Done k-nearest neighbor search!")
        return d, ind
