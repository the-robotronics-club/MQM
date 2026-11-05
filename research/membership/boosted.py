"""Gradient-boosted membership model factory."""


def create_model(random_seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=random_seed)
