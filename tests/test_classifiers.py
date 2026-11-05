from __future__ import annotations

import numpy as np

from inference.classifiers import PersonAttributeClassifiers, attribute_classifiers_enabled
from inference.zones import ZoneAssignment


def test_attribute_classifier_flag_is_explicit(monkeypatch):
    monkeypatch.setenv("ATTRIBUTE_CLASSIFIERS_ENABLED", "true")
    assert attribute_classifiers_enabled() is True
    monkeypatch.setenv("ATTRIBUTE_CLASSIFIERS_ENABLED", "false")
    assert attribute_classifiers_enabled() is False
    monkeypatch.setenv("ATTRIBUTE_CLASSIFIERS_ENABLED", "maybe")
    try:
        attribute_classifiers_enabled()
    except ValueError as exc:
        assert "true or false" in str(exc)
    else:
        raise AssertionError("invalid classifier flag must be rejected")


def test_reviewed_classifiers_load_and_classify_person_crops():
    classifiers = PersonAttributeClassifiers()
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    assignments = [
        ZoneAssignment(
            bbox=[10.0, 10.0, 100.0, 110.0],
            confidence=0.9,
            point=(55.0, 110.0),
        )
    ]
    attributes = classifiers.predict(image, assignments)
    assert len(attributes) == 1
    assert isinstance(attributes[0].is_queue, bool)
    assert isinstance(attributes[0].is_seated, bool)
    assert 0.0 <= attributes[0].queue_confidence <= 1.0
    assert 0.0 <= attributes[0].seated_confidence <= 1.0


def test_seated_only_path_does_not_require_queue_weights(monkeypatch):
    from inference.classifiers import SeatedClassifier

    monkeypatch.setenv("QUEUE_CLASSIFIER_WEIGHTS", "/missing/queue.pt")
    classifier = SeatedClassifier()
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    assignments = [
        ZoneAssignment(bbox=[10.0, 10.0, 100.0, 110.0], confidence=0.9, point=(55.0, 110.0))
    ]
    labels, probabilities = classifier.predict(image, assignments)
    assert labels[0] in (0, 1)
    assert 0 <= probabilities[0] <= 1
    assert classifier.predict(image, []) == ([], [])
