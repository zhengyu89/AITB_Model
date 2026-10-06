# Model smoke tests in Postman

1. Start the server from the repository root:
   ```bash
   uv run --locked uvicorn app.main:app --reload
   ```
2. Import `postman/model-tests.postman_collection.json` into Postman.
3. Open the collection's **Variables**. Set `base_url` to your server URL and `api_key` to the value of `API_KEY` in your `.env` if authentication is enabled. No real credentials are included in the collection.
4. Run the collection in order. Requests 02 and 03 use an embedded synthetic PNG, so no image setup is needed for those requests.
5. For request 04, open **Body > form-data** and select `postman/smoke-image.png` or a real photo for the `file` field. Use the Postman desktop app or desktop agent to access local files. Do not manually add a multipart Content-Type header. In the Collection Runner, allow local file access and select the file if needed.

The collection checks:

| Request | Expected result |
| --- | --- |
| 01 Health | HTTP 200 after startup has preloaded the models |
| 02 Base64 prediction | Both attraction and food classifier outputs, probabilities in [0, 1], embedding model metadata, reference candidates |
| 03 Embedding retrieval | Classification is null; candidate identities and similarity scores match request 02 for the same image |
| 04 File upload | Both classifier heads and embedding retrieval execute for a local image |
| 05 Invalid base64 | HTTP 400 |
| 06 Invalid topk | HTTP 422 |
| 07 Missing file | HTTP 422 |

The defaults expect `facebook/dinov2-large` with 1024 embedding dimensions, matching the supplied checkpoints. If you replace the checkpoints, update `expected_embedding_model` and `expected_embedding_dim`. Both classifier checks expect both supplied `.pth` files to be present.

Prediction requests require the configured Qdrant service and a populated collection of reference images. A 500 response may indicate Qdrant connectivity or collection problems, even if model preload succeeded. The nonempty-candidates assertion intentionally fails for an empty reference collection. Requests 02 and 03 should run consecutively without changing the image, `topk`, or reference data.

The synthetic image tests execution, not recognition accuracy. A `reject` prediction for it is valid. To evaluate recognition, replace `image_base64` with the encoded bytes of a labeled landmark/food photo, or upload that photo in request 04, and compare its class and retrieved matches with the known label. The API does not expose raw vectors; the collection checks embedding metadata and its use in retrieval rather than individual vector values.

To encode a photo for the JSON requests:

```bash
uv run --locked python -c "import base64; from pathlib import Path; print(base64.b64encode(Path('your-photo.jpg').read_bytes()).decode())"
```

Paste the output into the collection's `image_base64` variable. Keep `topk` between 1 and 20.

## Run the same requests with Python

With the API running, execute:

```bash
uv run --locked python scripts/test_model_api.py --base-url http://127.0.0.1:8000
```

This runner reads the collection, sends its seven requests using `httpx`, and checks the HTTP responses, classifier outputs, embedding metadata, and retrieval consistency. It loads `API_KEY` through the app's `.env` configuration. It prints the request details, full JSON response, elapsed time, and PASS/FAIL for each check with expected and actual values. It masks the API key and summarizes the base64 image bytes. These are Python checks; it does not execute Postman's JavaScript tests.

To save a detailed JSON report too:

```bash
uv run --locked python scripts/test_model_api.py --report /tmp/model-test-results.json
```
