"""Africa filter: real country outlines, since a bounding box cannot split Algeria from Spain."""
import numpy as np

from firms.region import in_africa


def test_african_cities_are_inside_and_european_ones_are_not():
    cities = {
        "Algiers": (36.75, 3.06, True), "Tunis": (36.80, 10.18, True), "Tangier": (35.77, -5.80, True),
        "Cairo": (30.04, 31.24, True), "Kinshasa": (-4.32, 15.31, True), "Dakar": (14.72, -17.47, True),
        "Madrid": (40.42, -3.70, False), "Seville": (37.39, -5.98, False), "Tarifa": (36.01, -5.60, False),
        "Palermo": (38.12, 13.36, False), "Athens": (37.98, 23.73, False), "Istanbul": (41.01, 28.98, False),
        "Las Palmas (Spain)": (28.12, -15.43, False), "Lisbon": (38.72, -9.14, False), "Tel Aviv": (32.08, 34.78, False),
    }
    got = in_africa([c[0] for c in cities.values()], [c[1] for c in cities.values()])
    assert {k: bool(v) for k, v in zip(cities, got)} == {k: c[2] for k, c in cities.items()}


def test_empty_and_array_inputs():
    assert in_africa([], []).shape == (0,)
    assert in_africa(np.array([36.75]), np.array([3.06])).tolist() == [True]
