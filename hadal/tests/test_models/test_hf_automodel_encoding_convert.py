import numpy
import torch

from hadal import HuggingfaceAutoModel
from hadal.tests.testing_tools import pytest_test_on_devices


class TestEncodeConvertToHuggingfaceAutoModel:
    def setup_method(self):
        self.model_name_or_path = "setu4993/LaBSE"
        self.model = HuggingfaceAutoModel(model_name_or_path=self.model_name_or_path)

    @pytest_test_on_devices
    def test_encode_convert_to_numpy(self, test_device):
        """Test encode() method with the convert_to="numpy"."""
        test_sentences = ["He likes dogs!", "She eats ice-cream."]
        embeddings = self.model.encode(sentences=test_sentences, device=test_device, convert_to="numpy")

        assert len(embeddings) == len(test_sentences)
        assert isinstance(embeddings, numpy.ndarray)

    @pytest_test_on_devices
    def test_encode_convert_to_torch(self, test_device):
        """Test encode() method with the convert_to="torch"."""
        test_sentences = ["He likes dogs!", "She eats ice-cream."]
        embeddings = self.model.encode(sentences=test_sentences, device=test_device, convert_to="torch")

        assert len(embeddings) == len(test_sentences)
        assert isinstance(embeddings, torch.Tensor)
