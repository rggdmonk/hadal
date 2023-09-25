from hadal import HuggingfaceAutoModel
from hadal.tests.testing_tools import pytest_test_on_devices


class TestEncodeNormHuggingfaceAutoModel:
    def setup_method(self):
        self.model_name_or_path = "setu4993/LaBSE"
        self.model = HuggingfaceAutoModel(model_name_or_path=self.model_name_or_path)

    @pytest_test_on_devices
    def test_encode_normalization_numpy(self, test_device):
        """Test if the embeddings are normalized correctly (numpy)."""
        test_sentences = ["I got bills I gotta pay", "So I'm gon' work, work, work every day"]

        not_normalized = self.model.encode(
            sentences=test_sentences,
            device=test_device,
            normalize_embeddings=False,
            convert_to="numpy",
        )
        normalized = self.model.encode(
            sentences=test_sentences,
            device=test_device,
            normalize_embeddings=True,
            convert_to="numpy",
        )

        assert len(not_normalized) == len(test_sentences)
        assert len(normalized) == len(test_sentences)
        assert not_normalized.all() == normalized.all()

    @pytest_test_on_devices
    def test_encode_normalization_torch(self, test_device):
        """Test if the embeddings are normalized correctly (torch)."""
        test_sentences = ["I got bills I gotta pay", "So I'm gon' work, work, work every day"]

        not_normalized = self.model.encode(
            sentences=test_sentences,
            device=test_device,
            normalize_embeddings=False,
            convert_to="torch",
        )
        normalized = self.model.encode(
            sentences=test_sentences,
            device=test_device,
            normalize_embeddings=True,
            convert_to="torch",
        )

        assert len(not_normalized) == len(test_sentences)
        assert len(normalized) == len(test_sentences)
        assert not_normalized.all() == normalized.all()
