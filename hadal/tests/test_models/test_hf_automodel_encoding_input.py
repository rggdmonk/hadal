import torch

from hadal import HuggingfaceAutoModel
from hadal.tests.testing_tools import pytest_test_on_devices


class TestEncodeInputHuggingfaceAutoModel:
    def setup_method(self):
        self.model_name_or_path = "setu4993/LaBSE"
        self.model = HuggingfaceAutoModel(model_name_or_path=self.model_name_or_path)

    @pytest_test_on_devices
    def test_encode_list_of_strings(self, test_device):
        """Test encode() method with the list of strings."""
        test_sentences: list[str] = ["One", "Two", "Three"]
        embeddings = self.model.encode(sentences=test_sentences, device=test_device)

        assert len(embeddings) == len(test_sentences)
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], torch.Tensor)
        assert embeddings[0].shape[0] == self.model.config.hidden_size

    @pytest_test_on_devices
    def test_encode_list_of_empty_strings(self, test_device):
        """Test encode() method with the list of empty strings."""
        test_sentences: list[str] = ["", ""]
        embeddings = self.model.encode(sentences=test_sentences, device=test_device)

        assert len(embeddings) == len(test_sentences)
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], torch.Tensor)
        assert embeddings[0].shape[0] == self.model.config.hidden_size

    @pytest_test_on_devices
    def test_encode_empty_list(self, test_device):
        """Test encode() method with the empty list."""
        test_sentences: list = []
        embeddings = self.model.encode(sentences=test_sentences, device=test_device)

        assert len(embeddings) == 0
        assert isinstance(embeddings, list)

    @pytest_test_on_devices
    def test_encode_string(self, test_device):
        """Test encode() method with the single string."""
        test_sentence: str = "One"
        embeddings = self.model.encode(sentences=test_sentence, device=test_device)

        assert len(embeddings) == 1
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], torch.Tensor)
        assert embeddings[0].shape[0] == self.model.config.hidden_size

    @pytest_test_on_devices
    def test_encode_empty_string(self, test_device):
        """Test encode() method with the empty string."""
        test_sentence: str = ""
        embeddings = self.model.encode(sentences=test_sentence, device=test_device)

        assert len(embeddings) == 1
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], torch.Tensor)
        assert embeddings[0].shape[0] == self.model.config.hidden_size
