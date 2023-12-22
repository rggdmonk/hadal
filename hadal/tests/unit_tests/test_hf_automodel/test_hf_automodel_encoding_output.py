from hadal import HuggingfaceAutoModel
from hadal.tests.testing_tools import pytest_test_on_devices


class TestEncodeOutputHuggingfaceAutoModel:
    def setup_method(self):
        self.model_name_or_path = "setu4993/LaBSE"
        self.model = HuggingfaceAutoModel(model_name_or_path=self.model_name_or_path)

    @pytest_test_on_devices
    def test_encode_output_value(self, test_device):
        """Test encode() method with the `pooler_output` and `last_hidden_state`."""
        test_sentences = ["He likes dogs!", "She eats ice-cream."]

        pooler_embeddings = self.model.encode(
            sentences=test_sentences,
            device=test_device,
            output_value="pooler_output",
            convert_to="torch",
        )
        last_hidden_embeddings = self.model.encode(
            sentences=test_sentences,
            device=test_device,
            output_value="last_hidden_state",
            convert_to="torch",
        )

        assert len(pooler_embeddings) == len(test_sentences)
        assert len(last_hidden_embeddings) == len(test_sentences)
        assert pooler_embeddings.all() == last_hidden_embeddings.all()
