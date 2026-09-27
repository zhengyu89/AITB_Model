import math
import unittest
from dataclasses import replace

from app.config import Settings
from app.services.geo_ranking import (
    GeoPoint,
    GeoPointError,
    accuracy_factor,
    age_factor,
    build_geo_point,
    geo_prior,
    geo_weight,
    rank_and_decide,
)

SETTINGS = replace(
    Settings(),
    accept_score=0.40,
    tentative_score=0.28,
    min_gap=0.03,
    geo_max_weight=0.08,
    geo_prior_distance_m=8000.0,
    geo_reorder_window=0.04,
)

MERSING_JETTY = (2.4316, 103.8388)
TIOMAN_TEKEK = (2.8194, 104.1597)
TIOMAN_SALANG = (2.8770, 104.1530)
TIOMAN_JUARA = (2.7920, 104.2050)
KULAI = (1.6561, 103.6032)


def place(name: str, score: float, coords: tuple[float, float] | None) -> dict:
    payload = {"display_name": name}
    if coords is not None:
        payload["location"] = {"lat": coords[0], "lon": coords[1]}
    return {
        "display_name": name,
        "class_path": f"attraction/{name}",
        "category": "attraction",
        "best_score": score,
        "total_score": score,
        "hit_count": 1,
        "payload": payload,
    }


def visual(*places: dict) -> list[dict]:
    return sorted(places, key=lambda item: item["best_score"], reverse=True)


def fresh_fix(coords: tuple[float, float]) -> GeoPoint:
    return GeoPoint(lat=coords[0], lon=coords[1], accuracy_m=10.0, age_s=5.0)


class GeoPriorTests(unittest.TestCase):
    def test_reference_values(self):
        self.assertAlmostEqual(geo_prior(0, 8000), 1.0)
        self.assertAlmostEqual(geo_prior(2000, 8000), 0.78, places=2)
        self.assertAlmostEqual(geo_prior(8000, 8000), 0.37, places=2)
        self.assertAlmostEqual(geo_prior(25000, 8000), 0.04, places=2)

    def test_monotonic(self):
        values = [geo_prior(d, 8000) for d in (0, 500, 5000, 50000)]
        self.assertEqual(values, sorted(values, reverse=True))


class GeoWeightTests(unittest.TestCase):
    def test_factors(self):
        self.assertEqual(accuracy_factor(50), 1.0)
        self.assertEqual(accuracy_factor(200), 0.5)
        self.assertEqual(accuracy_factor(201), 0.0)
        self.assertEqual(accuracy_factor(None), 0.5)
        self.assertEqual(age_factor(60), 1.0)
        self.assertEqual(age_factor(120), 0.5)
        self.assertEqual(age_factor(121), 0.0)
        self.assertEqual(age_factor(None), 0.5)

    def test_weight(self):
        self.assertAlmostEqual(geo_weight(fresh_fix(MERSING_JETTY), 0.08), 0.08)
        self.assertAlmostEqual(geo_weight(GeoPoint(1.0, 103.0, 100, 90), 0.08), 0.02)
        self.assertEqual(geo_weight(GeoPoint(1.0, 103.0, 800, 5), 0.08), 0.0)

    def test_legacy_point_has_low_weight(self):
        point = build_geo_point(None, None, legacy_lat=KULAI[0], legacy_lon=KULAI[1])
        self.assertTrue(point.legacy)
        self.assertAlmostEqual(geo_weight(point, 0.08), 0.02)


class BuildGeoPointTests(unittest.TestCase):
    def test_absent(self):
        self.assertIsNone(build_geo_point(None, None))

    def test_geo_fields_take_precedence_over_legacy(self):
        point = build_geo_point(2.0, 103.0, 10, 5, legacy_lat=1.0, legacy_lon=100.0)
        self.assertEqual((point.lat, point.lon, point.legacy), (2.0, 103.0, False))

    def test_rejects_malformed(self):
        bad = [
            dict(geo_lat=2.0, geo_lon=None),
            dict(geo_lat=None, geo_lon=103.0),
            dict(geo_lat=91.0, geo_lon=103.0),
            dict(geo_lat=2.0, geo_lon=181.0),
            dict(geo_lat=0.0, geo_lon=0.0),
            dict(geo_lat=math.nan, geo_lon=103.0),
            dict(geo_lat=2.0, geo_lon=math.inf),
            dict(geo_lat=2.0, geo_lon=103.0, geo_accuracy_m=-1),
            dict(geo_lat=2.0, geo_lon=103.0, geo_age_s=math.nan),
            dict(geo_lat=None, geo_lon=None, legacy_lat=2.0, legacy_lon=None),
        ]
        for fields in bad:
            with self.subTest(fields=fields), self.assertRaises(GeoPointError):
                build_geo_point(**fields)

    def test_error_message_has_no_coordinates(self):
        with self.assertRaises(GeoPointError) as ctx:
            build_geo_point(95.123456, 103.654321)
        self.assertNotIn("95.12", str(ctx.exception))


class RankAndDecideTests(unittest.TestCase):
    def test_edge_case_1_no_geo_accepts_tioman(self):
        grouped = visual(place("Tioman", 0.55, TIOMAN_TEKEK), place("Mersing Town", 0.30, MERSING_JETTY))
        result = rank_and_decide(grouped, None, SETTINGS)
        self.assertEqual(result.decision["status"], "accept")
        self.assertEqual(result.decision["row"]["display_name"], "Tioman")
        self.assertEqual((result.retrieval_scope, result.geo_reason, result.geo_weight), ("global", "not_provided", 0.0))

    def test_old_bug_kulai_legacy_fix_does_not_pick_nearest(self):
        grouped = visual(place("Tioman", 0.55, TIOMAN_TEKEK), place("Mersing Town", 0.30, MERSING_JETTY))
        point = build_geo_point(None, None, legacy_lat=KULAI[0], legacy_lon=KULAI[1])
        result = rank_and_decide(grouped, point, SETTINGS)
        self.assertEqual(result.decision["status"], "accept")
        self.assertEqual(result.decision["row"]["display_name"], "Tioman")

    def test_edge_case_4_strong_visual_match_is_locked(self):
        grouped = visual(place("Tioman", 0.60, TIOMAN_TEKEK), place("Mersing Jetty", 0.35, MERSING_JETTY))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertEqual(result.decision["status"], "accept")
        self.assertEqual(result.decision["row"]["display_name"], "Tioman")
        self.assertEqual(result.geo_reason, "strong_visual_match")
        self.assertEqual(result.retrieval_scope, "geo_reranked")
        # Gap stays the visual gap; geo cannot erode it.
        self.assertAlmostEqual(result.decision["gap"], 0.25)

    def test_geo_disambiguates_two_similar_beaches(self):
        grouped = visual(place("Juara Beach", 0.45, TIOMAN_JUARA), place("Salang Beach", 0.44, TIOMAN_SALANG))
        self.assertEqual(rank_and_decide(grouped, None, SETTINGS).decision["status"], "tentative")

        result = rank_and_decide(grouped, fresh_fix(TIOMAN_SALANG), SETTINGS)
        self.assertEqual(result.decision["status"], "accept")
        self.assertEqual(result.decision["row"]["display_name"], "Salang Beach")
        self.assertEqual(result.geo_reason, "geo_disambiguated")
        self.assertEqual(result.reorder_window_size, 2)

    def test_reorder_window_keeps_far_visual_leader(self):
        grouped = visual(place("Tioman", 0.39, TIOMAN_TEKEK), place("Mersing Jetty", 0.32, MERSING_JETTY))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertEqual(result.decision["status"], "tentative")
        self.assertEqual(result.decision["row"]["display_name"], "Tioman")
        self.assertEqual(result.order[1]["display_name"], "Mersing Jetty")
        self.assertEqual(result.geo_reason, "strong_visual_match")

    def test_reordered_within_window_without_accept(self):
        grouped = visual(place("Far Beach", 0.36, TIOMAN_TEKEK), place("Near Beach", 0.34, MERSING_JETTY))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertEqual(result.decision["status"], "tentative")
        self.assertEqual(result.decision["row"]["display_name"], "Near Beach")
        self.assertEqual(result.geo_reason, "reordered")

    def test_no_effect_when_leader_is_also_nearest(self):
        grouped = visual(place("Near Beach", 0.36, MERSING_JETTY), place("Far Beach", 0.34, TIOMAN_TEKEK))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertEqual(result.decision["row"]["display_name"], "Near Beach")
        self.assertEqual(result.geo_reason, "no_effect")

    def test_sub_threshold_candidate_never_promoted(self):
        grouped = visual(place("Far Beach", 0.30, TIOMAN_TEKEK), place("Near Beach", 0.27, MERSING_JETTY))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertEqual(result.decision["status"], "tentative")
        self.assertEqual(result.decision["row"]["display_name"], "Far Beach")

    def test_weak_match_stays_rejected(self):
        grouped = visual(place("Far Beach", 0.27, TIOMAN_TEKEK), place("Near Beach", 0.26, MERSING_JETTY))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertEqual(result.decision["status"], "reject")

    def test_geo_cannot_lift_below_accept_threshold(self):
        grouped = visual(place("Near Beach", 0.39, MERSING_JETTY), place("Far Beach", 0.36, TIOMAN_TEKEK))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertEqual(result.decision["status"], "tentative")

    def test_places_without_coordinates_are_kept(self):
        grouped = visual(place("Unknown Spot", 0.45, None), place("Near Beach", 0.44, MERSING_JETTY))
        result = rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        names = [item["display_name"] for item in result.order]
        self.assertEqual(sorted(names), ["Near Beach", "Unknown Spot"])
        unknown = next(item for item in result.order if item["display_name"] == "Unknown Spot")
        self.assertIsNone(unknown["distance_m"])
        self.assertAlmostEqual(unknown["combined_score"], 0.45)

    def test_zero_weight_falls_back_to_global(self):
        grouped = visual(place("Far Beach", 0.36, TIOMAN_TEKEK), place("Near Beach", 0.34, MERSING_JETTY))
        point = GeoPoint(MERSING_JETTY[0], MERSING_JETTY[1], accuracy_m=800, age_s=5)
        result = rank_and_decide(grouped, point, SETTINGS)
        self.assertEqual((result.retrieval_scope, result.geo_reason), ("global", "zero_weight"))
        self.assertNotIn("distance_m", result.order[0])

    def test_empty_candidates(self):
        self.assertEqual(rank_and_decide([], None, SETTINGS).decision["status"], "reject")
        self.assertEqual(rank_and_decide([], fresh_fix(MERSING_JETTY), SETTINGS).decision["status"], "reject")

    def test_input_not_mutated(self):
        grouped = visual(place("Far Beach", 0.36, TIOMAN_TEKEK), place("Near Beach", 0.34, MERSING_JETTY))
        rank_and_decide(grouped, fresh_fix(MERSING_JETTY), SETTINGS)
        self.assertNotIn("combined_score", grouped[0])


if __name__ == "__main__":
    unittest.main()
