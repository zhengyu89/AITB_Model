import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import embeddings as embeddings_routes
from app.services.text_embedder import TextEncoderUnavailable


class EmbeddingsRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(embeddings_routes.router)
        self.client = TestClient(app)
        patcher = mock.patch.object(
            embeddings_routes,
            "encode_texts",
            side_effect=lambda texts: [[float(len(t)), 0.0, 1.0] for t in texts],
        )
        self.encode_texts = patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_one_vector_per_text_in_order(self):
        response = self.client.post("/embeddings", json={"texts": ["cendol", "nasi lemak"]})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["model"], "all-MiniLM-L6-v2")
        self.assertEqual(body["dimension"], 3)
        self.assertEqual(len(body["embeddings"]), 2)
        self.assertEqual(body["embeddings"][0][0], 6.0)
        self.assertEqual(body["embeddings"][1][0], 10.0)

    def test_explicit_matching_model_is_accepted(self):
        response = self.client.post(
            "/embeddings", json={"texts": ["cendol"], "model": "all-MiniLM-L6-v2"}
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_unsupported_model_is_rejected(self):
        response = self.client.post(
            "/embeddings", json={"texts": ["cendol"], "model": "text-embedding-3-small"}
        )
        self.assertEqual(response.status_code, 400)
        self.encode_texts.assert_not_called()

    def test_empty_texts_is_rejected(self):
        response = self.client.post("/embeddings", json={"texts": []})
        self.assertEqual(response.status_code, 422)
        self.encode_texts.assert_not_called()

    def test_over_max_batch_is_rejected(self):
        response = self.client.post("/embeddings", json={"texts": ["x"] * 129})
        self.assertEqual(response.status_code, 422)
        self.encode_texts.assert_not_called()

    def test_max_batch_is_accepted(self):
        response = self.client.post("/embeddings", json={"texts": ["x"] * 128})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["embeddings"]), 128)

    def test_missing_texts_field_is_rejected(self):
        response = self.client.post("/embeddings", json={})
        self.assertEqual(response.status_code, 422)

    def test_model_unavailable_returns_503(self):
        self.encode_texts.side_effect = TextEncoderUnavailable("model not loaded")
        response = self.client.post("/embeddings", json={"texts": ["cendol"]})
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
