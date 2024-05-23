from dataclasses import dataclass


@dataclass
class Files:
    data: str
    metadata: str


@dataclass
class Paths:
    data: str


@dataclass
class GuideHelperConfig:
    files: Files
    paths: Paths
