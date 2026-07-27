import pytest
from attack_core import ATTACKIndex, ATTACKLoader

from attack_mapping.enricher import ATTACKEnricher


@pytest.fixture
def enricher():
    loader = ATTACKLoader()
    index = ATTACKIndex(loader)
    return ATTACKEnricher(index)


def mapped_ids(mappings):
    return {m.subtechnique_id or m.technique_id for m in mappings}


class TestModelPrivacyEnricher:
    def test_membership_inference(self, enricher):
        mappings = enricher.enrich("membership_inference_success", {"confidence": 0.9})
        assert "T1005" in mapped_ids(mappings)
        assert "T1213.002" in mapped_ids(mappings)

    def test_model_stealing(self, enricher):
        mappings = enricher.enrich("model_stealing_detected", {"confidence": 0.85})
        assert "T1005" in mapped_ids(mappings)
        assert "T1114" in mapped_ids(mappings)

    def test_differential_privacy_bypass(self, enricher):
        mappings = enricher.enrich("differential_privacy_bypass", {"confidence": 0.8})
        assert "T1685" in mapped_ids(mappings)
        assert "T1565" in mapped_ids(mappings)
