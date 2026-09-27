"""GPS-aware re-ranking of grouped visual candidates.

Device GPS is treated as a soft prior, never a filter:

- Retrieval is always global; this module only reorders places that are already
  visually close to the visual #1 (the reorder window).
- Acceptance thresholds are always checked against raw visual similarity, so
  GPS cannot create a match out of a weak one.
- Places without coordinates stay in the list with a prior of 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import Settings
from app.services.qdrant_retrieval import haversine_distance_m

# Tolerance for float comparisons against the reorder window.
_EPS = 1e-9

# Accuracy (metres) and age (seconds) breakpoints for the weight factors.
_ACCURACY_FULL_M = 50.0
_ACCURACY_HALF_M = 200.0
_AGE_FULL_S = 60.0
_AGE_HALF_S = 120.0
# Factor used when accuracy or age is unknown (legacy callers).
_UNKNOWN_FACTOR = 0.5


class GeoPointError(ValueError):
    """Raised when the supplied geo fields are malformed."""


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    accuracy_m: float | None = None
    age_s: float | None = None
    legacy: bool = False


@dataclass
class RankingResult:
    order: list[dict]
    decision: dict
    retrieval_scope: str
    geo_reason: str
    geo_weight: float
    reorder_window_size: int


def build_geo_point(
    geo_lat: float | None,
    geo_lon: float | None,
    geo_accuracy_m: float | None = None,
    geo_age_s: float | None = None,
    legacy_lat: float | None = None,
    legacy_lon: float | None = None,
) -> GeoPoint | None:
    """Validate the request's geo fields and return a GeoPoint, or None when absent.

    `geo_lat`/`geo_lon` take precedence. Legacy `user_lat`/`user_lon` are accepted
    as a point with unknown accuracy and age, which gives them a low weight.
    Error messages never include the submitted coordinate values.
    """
    if geo_lat is not None or geo_lon is not None:
        lat, lon = _validate_pair(geo_lat, geo_lon, "geo_lat", "geo_lon")
        return GeoPoint(
            lat=lat,
            lon=lon,
            accuracy_m=_validate_non_negative(geo_accuracy_m, "geo_accuracy_m"),
            age_s=_validate_non_negative(geo_age_s, "geo_age_s"),
        )

    if legacy_lat is not None or legacy_lon is not None:
        lat, lon = _validate_pair(legacy_lat, legacy_lon, "user_lat", "user_lon")
        return GeoPoint(lat=lat, lon=lon, legacy=True)

    return None


def _validate_pair(lat: float | None, lon: float | None, lat_name: str, lon_name: str) -> tuple[float, float]:
    if lat is None or lon is None:
        raise GeoPointError(f"{lat_name} and {lon_name} must be supplied together.")
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise GeoPointError(f"{lat_name} and {lon_name} must be finite numbers.")
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise GeoPointError(f"{lat_name} or {lon_name} is out of range.")
    if lat == 0.0 and lon == 0.0:
        raise GeoPointError(f"{lat_name}/{lon_name} of (0, 0) is not a valid fix.")
    return lat, lon


def _validate_non_negative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise GeoPointError(f"{name} must be a finite number >= 0.")
    return value


def geo_prior(distance_m: float, scale_m: float) -> float:
    """Smooth distance prior: 1.0 at the user, decaying exponentially with distance."""
    return math.exp(-max(distance_m, 0.0) / scale_m)


def accuracy_factor(accuracy_m: float | None) -> float:
    if accuracy_m is None:
        return _UNKNOWN_FACTOR
    if accuracy_m <= _ACCURACY_FULL_M:
        return 1.0
    if accuracy_m <= _ACCURACY_HALF_M:
        return 0.5
    return 0.0


def age_factor(age_s: float | None) -> float:
    if age_s is None:
        return _UNKNOWN_FACTOR
    if age_s <= _AGE_FULL_S:
        return 1.0
    if age_s <= _AGE_HALF_S:
        return 0.5
    return 0.0


def geo_weight(point: GeoPoint, max_weight: float) -> float:
    return max_weight * accuracy_factor(point.accuracy_m) * age_factor(point.age_s)


def place_coordinates(payload: dict) -> tuple[float, float] | None:
    """Read (lat, lon) from a Qdrant payload's `location` object or top-level fields."""
    location = payload.get("location")
    if isinstance(location, dict):
        lat, lon = location.get("lat", payload.get("lat")), location.get("lon", payload.get("lon"))
    else:
        lat, lon = payload.get("lat"), payload.get("lon")
    if lat is None or lon is None:
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return lat, lon


def rank_and_decide(grouped: list[dict], point: GeoPoint | None, settings: Settings) -> RankingResult:
    """Order grouped places and make the accept/tentative/reject decision.

    `grouped` must already be sorted by visual similarity (`best_score`) descending,
    as returned by `aggregate_qdrant_results`. Input dicts are not mutated.
    """
    visual = list(grouped)

    if point is None:
        return _visual_result(visual, settings, geo_reason="not_provided")

    weight = geo_weight(point, settings.geo_max_weight)
    if weight <= 0.0:
        return _visual_result(visual, settings, geo_reason="zero_weight")

    scored = [_with_geo_score(place, point, weight, settings.geo_prior_distance_m) for place in visual]
    if not scored:
        return RankingResult(
            order=[],
            decision=_decide(None, None, settings),
            retrieval_scope="geo_reranked",
            geo_reason="no_effect",
            geo_weight=weight,
            reorder_window_size=0,
        )

    window = _reorder_window(scored, settings)
    if len(window) == 1:
        # Visual #1 is clearly ahead: geo must not touch the order or the gap.
        order = scored
        decision = _decide(order[0], _similarity_gap(order), settings)
        return RankingResult(
            order=order,
            decision=decision,
            retrieval_scope="geo_reranked",
            geo_reason="strong_visual_match",
            geo_weight=weight,
            reorder_window_size=1,
        )

    reordered = sorted(
        window,
        key=lambda place: (place["combined_score"], _similarity(place)),
        reverse=True,
    )
    order = reordered + scored[len(window):]
    combined_gap = order[0]["combined_score"] - order[1]["combined_score"]
    decision = _decide(order[0], combined_gap, settings)

    visual_decision = _decide(scored[0], _similarity_gap(scored), settings)
    top_changed = _place_key(order[0]) != _place_key(scored[0])
    if decision["status"] == "accept" and visual_decision["status"] != "accept":
        reason = "geo_disambiguated"
    elif top_changed:
        reason = "reordered"
    else:
        reason = "no_effect"

    return RankingResult(
        order=order,
        decision=decision,
        retrieval_scope="geo_reranked",
        geo_reason=reason,
        geo_weight=weight,
        reorder_window_size=len(window),
    )


def _visual_result(visual: list[dict], settings: Settings, geo_reason: str) -> RankingResult:
    top = visual[0] if visual else None
    return RankingResult(
        order=visual,
        decision=_decide(top, _similarity_gap(visual), settings),
        retrieval_scope="global",
        geo_reason=geo_reason,
        geo_weight=0.0,
        reorder_window_size=0,
    )


def _reorder_window(scored: list[dict], settings: Settings) -> list[dict]:
    """Places GPS may reorder: within the window of visual #1 and raw-viable.

    A place below `tentative_score` is never promoted over the visual #1, so geo
    cannot turn a tentative result into a reject either.
    """
    top_similarity = _similarity(scored[0])
    window = [scored[0]]
    for place in scored[1:]:
        similarity = _similarity(place)
        if top_similarity - similarity > settings.geo_reorder_window + _EPS:
            break
        if similarity < settings.tentative_score:
            break
        window.append(place)
    return window


def _with_geo_score(place: dict, point: GeoPoint, weight: float, scale_m: float) -> dict:
    scored = dict(place)
    coordinates = place_coordinates(place.get("payload") or {})
    if coordinates is None:
        scored["distance_m"] = None
        prior = 0.0
    else:
        distance_m = haversine_distance_m(point.lat, point.lon, coordinates[0], coordinates[1])
        scored["distance_m"] = distance_m
        prior = geo_prior(distance_m, scale_m)
    scored["geo_prior"] = prior
    scored["combined_score"] = _similarity(place) + weight * prior
    return scored


def _decide(top: dict | None, gap: float | None, settings: Settings) -> dict:
    """Accept/tentative/reject using raw similarity for thresholds."""
    if top is not None:
        score = _similarity(top)
        gap = gap if gap is not None else score
        if score >= settings.accept_score and gap >= settings.min_gap:
            return _decision("accept", top, score, gap)
        if score >= settings.tentative_score:
            return _decision("tentative", top, score, gap)
    return {"status": "reject", "row": None, "score": None, "gap": None, "hit_count": 0}


def _decision(status: str, top: dict, score: float, gap: float) -> dict:
    return {
        "status": status,
        "row": top,
        "score": score,
        "gap": gap,
        "hit_count": int(top.get("hit_count") or 0),
    }


def _similarity_gap(order: list[dict]) -> float | None:
    if not order:
        return None
    second = _similarity(order[1]) if len(order) > 1 else 0.0
    return _similarity(order[0]) - second


def _similarity(place: dict) -> float:
    return float(place.get("best_score") or 0.0)


def _place_key(place: dict) -> str:
    return str(place.get("class_path") or place.get("display_name") or "")
