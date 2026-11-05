"""Logistic Regression membership model factory."""


def create_model(random_seed: int = 42):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed)
