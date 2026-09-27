# Provenance

Tracks which parts of this repo are our own code, adapted from a reference, or reused as-is.

| File / component | Status | Source |
|---|---|---|
| resumable_extract.py | Adapted | AIML-MED/CAPE `datasets.py::_extract()` — added checkpointing every 500 images so extraction survives interruptions |
| experiments/01_sanity_check.ipynb | Written by us | — |
| src/explain.py | Adapted | CAM and CAPE maps come directly from AIML-MED/CAPE's model.py forward pass. Grad-CAM and Grad-CAM++ are our own implementation (standard Selvaraju et al. / Chattopadhay et al. formulas — no reference eval script existed in AIML-MED/CAPE to adapt from) |
| src/metrics.py | Written by us | Implemented from the paper's Section 4.3 equations directly, since AIML-MED/CAPE's repo does not include an evaluation/metrics script |