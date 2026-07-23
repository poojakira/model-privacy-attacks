from attack_core import ATTACKLoader, ATTACKIndex
from attack_mapping.enricher import ATTACKEnricher
from attack_mapping.reporter import NavigatorLayerReporter

loader = ATTACKLoader()
index = ATTACKIndex(loader)
enricher = ATTACKEnricher(index)
reporter = NavigatorLayerReporter()

all_mappings = []
for ft in ['membership_inference_success', 'model_stealing_detected', 'attribute_inference', 'gradient_leakage', 'model_inversion_pii', 'differential_privacy_bypass', 'federated_learning_poisoning', 'api_probing_extraction']:
    mappings = enricher.enrich(ft, {'confidence': 0.8})
    all_mappings.extend(mappings)

layer = reporter.generate('model-privacy-attacks', all_mappings)
import json
data = json.loads(layer)
print(f'Techniques mapped: {len(data["techniques"])}')
for t in data['techniques']:
    print(f'  {t["techniqueID"]}: score={t["score"]}')