import base64
import io
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.routes import predict as predict_routes
from app.services.geo_ranking import rank_and_decide
from app.config import get_settings

MERSING_JETTY = {"lat": 2.4316, "lon": 103.8388}
TIOMAN = {"lat": 2.8194, "lon": 104.1597}


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_predict_image(pil_image, topk, geo_point=None, include_classification=True, include_debug=False):
    """Stand-in for the model pipeline that exercises the real ranking and response builders."""
    from app.services import pipeline

    grouped = [
        {"display_name": "Tioman", "class_path": "a/tioman", "best_score": 0.60, "hit_count": 3,
         "payload": {"location": TIOMAN}},
        {"display_name": "Mersing Jetty", "class_path": "a/jetty", "best_score": 0.35, "hit_count": 1,
         "payload": {"location": MERSING_JETTY}},
    ]
    ranking = rank_and_decide(grouped, geo_point, get_settings())
    geo_applied = ranking.retrieval_scope == "geo_reranked"
    return {
        "status": ranking.decision["status"],
        "retrieval_scope": ranking.retrieval_scope,
        "geo_reason": ranking.geo_reason,
        "geo_weight": ranking.geo_weight,
        "final_match": pipeline._build_final_match(ranking.decision, geo_applied),
        "candidates": pipeline._build_candidates(ranking.order[:topk], geo_applied),
        "classification": None,
    }


class PredictRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(predict_routes.router)
        self.client = TestClient(app)
        patcher = mock.patch.object(predict_routes, "predict_image", side_effect=_fake_predict_image)
        self.predict_image = patcher.start()
        self.addCleanup(patcher.stop)

    def _upload(self, data=None, params=None):
        return self.client.post(
            "/predict/upload",
            files={"file": ("x.png", _png_bytes(), "image/png")},
            data=data or {},
            params=params or {},
        )

    def test_upload_without_geo_is_global_and_has_no_distance(self):
        response = self._upload()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual((body["retrieval_scope"], body["geo_reason"], body["geo_weight"]), ("global", "not_provided", 0.0))
        self.assertIsNone(body["final_match"]["distance_m"])
        self.assertTrue(all(c["distance_m"] is None and c["combined_score"] is None for c in body["candidates"]))

    def test_upload_reads_geo_from_form_body(self):
        response = self._upload(data={
            "geo_lat": MERSING_JETTY["lat"], "geo_lon": MERSING_JETTY["lon"],
            "geo_accuracy_m": 10, "geo_age_s": 5,
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["retrieval_scope"], "geo_reranked")
        self.assertEqual(body["geo_reason"], "strong_visual_match")
        self.assertEqual(body["final_match"]["name"], "Tioman")
        self.assertGreater(body["final_match"]["distance_m"], 40_000)
        geo_point = self.predict_image.call_args.kwargs["geo_point"]
        self.assertFalse(geo_point.legacy)
        self.assertEqual(geo_point.accuracy_m, 10)

    def test_upload_accepts_legacy_query_params(self):
        response = self._upload(params={"user_lat": MERSING_JETTY["lat"], "user_lon": MERSING_JETTY["lon"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(response.json()["geo_weight"], 0.02)
        self.assertTrue(self.predict_image.call_args.kwargs["geo_point"].legacy)

    def test_upload_rejects_half_pair(self):
        response = self._upload(data={"geo_lat": 2.43})
        self.assertEqual(response.status_code, 422)
        self.predict_image.assert_not_called()

    def test_json_endpoint_reads_geo_fields(self):
        response = self.client.post("/predict/", json={
            "image_base64": base64.b64encode(_png_bytes()).decode(),
            "geo_lat": MERSING_JETTY["lat"], "geo_lon": MERSING_JETTY["lon"],
            "geo_accuracy_m": 10, "geo_age_s": 5,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["retrieval_scope"], "geo_reranked")

    def test_json_endpoint_rejects_zero_zero(self):
        response = self.client.post("/predict/", json={
            "image_base64": base64.b64encode(_png_bytes()).decode(),
            "geo_lat": 0, "geo_lon": 0,
        })
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
