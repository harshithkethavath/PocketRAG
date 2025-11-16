from dataclasses import dataclass


@dataclass
class PocketRAGConfig:
    """
    Global configuration for PocketRAG runs.

    This will grow over time to include:
    - dataset paths
    - retrieval settings
    - generation model name
    - evaluation options
    """
    device_pref: str = "auto"