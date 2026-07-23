"""
ATT&CK Enricher for model-privacy-attacks.
"""
from attack_core.index import ATTACKIndex
from attack_core.models import ATTACKMapping
from typing import List, Dict, Any


class ATTACKEnricher:
    def __init__(self, index: ATTACKIndex):
        self.index = index
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

    def enrich(self, finding_type: str, metadata: Dict[str, Any]) -> List[ATTACKMapping]:
        technique_ids = self._rule_table.get(finding_type, [])
        mappings = []
        for tid in technique_ids:
            tech = self.index.get(tid)
            if tech:
                tactic = self.index._tactics.get(tech.tactic_ids[0] if tech.tactic_ids else "", None)
                mappings.append(ATTACKMapping(
                    tactic_id=tech.tactic_ids[0] if tech.tactic_ids else "unknown",
                    tactic_name=tactic.name if tactic else "unknown",
                    technique_id=tech.attack_id,
                    technique_name=tech.name,
                    subtechnique_id=tech.attack_id if tech.is_subtechnique else None,
                    subtechnique_name=tech.name if tech.is_subtechnique else None,
                    domain=tech.domain,
                    confidence=metadata.get("confidence", 0.5),
                    data_sources=tech.data_sources,
                    platforms=tech.platforms,
                    url=tech.url,
                ))
        return mappings