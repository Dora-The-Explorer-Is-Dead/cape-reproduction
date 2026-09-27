# Provenance

Tracks which parts of this repo are our own code, adapted from a reference, or reused as-is.

| File / component | Status | Source |
|---|---|---|
| resumable_extract.py | Adapted | AIML-MED/CAPE `datasets.py::_extract()` — added checkpointing every 500 images so extraction survives interruptions |
| experiments/01_sanity_check.ipynb | Written by us | — |