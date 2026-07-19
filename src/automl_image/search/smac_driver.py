from __future__ import annotations


class RandomSearcher:
    def __init__(self, seed: int = 0) -> None:
        raise NotImplementedError


class SMACHyperbandSearcher:
    def __init__(self, seed: int = 0) -> None:
        raise NotImplementedError


class SMACBOHBSearcher:
    def __init__(self, seed: int = 0, cost_aware: bool = True) -> None:
        raise NotImplementedError
