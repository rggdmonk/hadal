import numpy

import hadal
from hadal.tests.testing_tools import pytest_test_on_devices


@pytest_test_on_devices
def test_minimal_margin_based(test_device):
    """Test minimal example for MarginBasedPipeline."""
    model_name = "setu4993/LaBSE"

    source_test = [
        "I eat a random word every year.",  # corrupted sentence [2]
        "No",  # not in target sentences
        "I like wine but I don't have it.",
        "Every day, he rides his bike to work.",
    ]

    target_test = [
        "Me gusta el vino pero no lo tengo.",
        "Cada día, pedalea con su bici hasta el trabajo.",
        "Ella come una manzana todos los días.",
        "En la actualidad, los manzanos sólo empiezan a florecer en abril.",
    ]

    alignment_config = hadal.MarginBasedPipeline(model_name_or_path=model_name, model_device=test_device)

    result = alignment_config.make_alignments(
        source_sentences=source_test,
        target_sentences=target_test,
        knn_neighbors=2,
    )

    expected_len = 3

    assert len(result) == expected_len
    assert type(result[0][0]) == numpy.float64
    assert result[0][1] == source_test[2]
    assert result[1][1] == source_test[3]
    assert result[2][1] == source_test[0]
