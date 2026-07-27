"""
ATT&CK Enricher for model-privacy-attacks.
"""

from typing import Any

from attack_core.index import ATTACKIndex
from attack_core.mapping import ATTACKMappingBuilder
from attack_core.models import ATTACKMapping


class ATTACKEnricher:
    def __init__(self, index: ATTACKIndex):
        self.index = index
        self.mapping_builder = ATTACKMappingBuilder(index)
        self._rule_table = {
            "membership_inference_success": ["T1005", "T1213.002"],
            "model_stealing_detected": ["T1005", "T1114"],
            "attribute_inference": ["T1552", "T1213"],
            "gradient_leakage": ["T1005", "T1557"],
            "model_inversion_pii": ["T1005", "T1078"],
            "differential_privacy_bypass": ["T1685", "T1565"],
            "federated_learning_poisoning": ["T1195", "T1565"],
            "api_probing_extraction": ["T1190", "T1595"],
        }

    def enrich(self, finding_type: str, metadata: dict[str, Any]) -> list[ATTACKMapping]:
        confidence = metadata.get("confidence", 0.5)
        technique_ids = self._rule_table.get(finding_type, [])
        return self.mapping_builder.build_many(technique_ids, confidence)
