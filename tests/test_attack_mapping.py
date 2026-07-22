import pytest
from attack_core import ATTACKLoader, ATTACKIndex
from attack_mapping.enricher import ATTACKEnricher


@pytest.fixture
def enricher():
    loader = ATTACKLoader()
    index = ATTACKIndex(loader)
    return ATTACKEnricher(index)


class TestModelPrivacyEnricher:
    def test_membership_inference(self, enricher):
        mappings = enricher.enrich("membership_inference_success", {"confidence": 0.9})
        technique_ids = [m.technique_id for m in mappings]
        assert "T1005" in technique_ids
        assert "T1213.002" in technique_ids

    def test_model_stealing(self, enricher):
        mappings = enricher.enrich("model_stealing_detected", {"confidence": 0.85})
        technique_ids = [m.technique_id for m in mappings]
        assert "T1005" in technique_ids
        assert "T1114" in technique_ids

    def test_differential_privacy_bypass(self, enricher):
        mappings = enricher.enrich("differential_privacy_bypass", {"confidence": 0.8})
        technique_ids = [m.technique_id for m in mappings]
        assert "T1562" in technique_ids
        assert "T1565" in technique_ids