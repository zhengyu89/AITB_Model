"""Send model smoke tests and display each request, response, and check."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import get_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--report', type=Path, help='Also save detailed results as JSON')
    options = parser.parse_args()
    collection = json.loads((ROOT / 'postman/model-tests.postman_collection.json').read_text())
    variables = {v['key']: v['value'] for v in collection['variable']}
    variables.update(base_url=options.base_url.rstrip('/'), api_key=get_settings().api_key or '')
    results = []
    baseline = None

    def redact(text):
        key = variables['api_key']
        return text.replace(key, '[REDACTED]') if key else text

    def display(value):
        print(redact(value), flush=True)

    def expand(text):
        for key, value in variables.items():
            text = text.replace('{{' + key + '}}', value)
        return text

    def finite(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

    with httpx.Client(timeout=180) as client:
        for index, item in enumerate(collection['item']):
            request = item['request']
            url = expand(request['url'])
            expected = [200, 200, 200, 200, 400, 422, 422][index]
            headers = {h['key']: expand(h['value']) for h in request['header']}
            args = {'headers': headers}
            body = request.get('body', {})
            request_body = None
            if body.get('mode') == 'raw':
                args['content'] = expand(body['raw'])
                request_body = json.loads(args['content'])
                # Keep console/report readable; the exact fixture is in the collection.
                image = request_body.get('image_base64', '')
                if len(image) > 100:
                    request_body['image_base64'] = f'[embedded image: {len(image)} base64 characters]'
            if body.get('mode') == 'formdata':
                image = (ROOT / 'postman/smoke-image.png').read_bytes()
                args['files'] = {'file': ('smoke-image.png', image, 'image/png')}
                request_body = {'file': 'postman/smoke-image.png', 'bytes': len(image), 'content_type': 'image/png'}
            result = {
                'name': item['name'],
                'request': {'method': request['method'], 'url': url,
                            'headers': {k: '[REDACTED]' if k.lower() == 'x-api-key' else v for k, v in headers.items()},
                            'body': request_body},
                'expected_http_status': expected, 'checks': [],
            }
            results.append(result)
            display('\n' + '=' * 72 + '\n' + item['name'])
            display(json.dumps(result['request'], indent=2))

            def check(name, passed, actual, expected_value):
                entry = {'name': name, 'passed': bool(passed), 'expected': expected_value, 'actual': actual}
                result['checks'].append(entry)
                display(f"{'PASS' if passed else 'FAIL'} | {name} | expected={expected_value!r} | actual={actual!r}")

            started = time.perf_counter()
            try:
                response = client.request(request['method'], url, **args)
            except httpx.RequestError as exc:
                result['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 2)
                check('HTTP request completed', False, str(exc), 'reachable API')
                continue
            result['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 2)
            result['http_status'] = response.status_code
            try:
                data = response.json()
            except ValueError:
                data = response.text
            result['response'] = data
            display(f"Response: HTTP {response.status_code}, {result['elapsed_ms']} ms")
            display(json.dumps(data, indent=2, ensure_ascii=False))
            check('HTTP status', response.status_code == expected, response.status_code, expected)
            check('JSON object response', isinstance(data, dict), type(data).__name__, 'dict')
            if response.status_code != expected or not isinstance(data, dict):
                continue
            if index == 0:
                check('Health status', data.get('status') == 'ok', data.get('status'), 'ok')
            elif index in (1, 2, 3):
                debug = data.get('debug') or {}
                check('Embedding model', debug.get('embedding_model') == variables['expected_embedding_model'], debug.get('embedding_model'), variables['expected_embedding_model'])
                check('Embedding dimension', debug.get('embedding_dim') == int(variables['expected_embedding_dim']), debug.get('embedding_dim'), int(variables['expected_embedding_dim']))
                check('Device', debug.get('device') in ('cpu', 'cuda'), debug.get('device'), 'cpu or cuda')
                check('Decision status', data.get('status') in ('accept', 'tentative', 'reject'), data.get('status'), 'accept, tentative, or reject')
                check('Global retrieval scope', data.get('retrieval_scope') == 'global', data.get('retrieval_scope'), 'global')
                match = data.get('final_match')
                valid_match = match is None if data.get('status') == 'reject' else isinstance(match, dict)
                check('Final match agrees with decision', valid_match, type(match).__name__, 'null for reject; object otherwise')
                candidates = data.get('candidates')
                check('Candidate array', isinstance(candidates, list), type(candidates).__name__, 'list')
                if not isinstance(candidates, list):
                    candidates = []
                check('Reference candidate count', 0 < len(candidates) <= int(variables['topk']), len(candidates), f"1..{variables['topk']}")
                for n, candidate in enumerate(candidates, 1):
                    score = candidate.get('similarity')
                    hits = candidate.get('reference_hits')
                    check(f'Candidate {n} similarity is finite', finite(score), score, 'finite number')
                    check(f'Candidate {n} reference hits', isinstance(hits, int) and hits >= 1, hits, 'integer >= 1')
                classification = data.get('classification')
                if index in (1, 3):
                    for branch in ('attraction_top1', 'food_top1'):
                        prediction = (classification or {}).get(branch) or {}
                        path = prediction.get('class_path')
                        probability = prediction.get('probability')
                        check(f'{branch} class', isinstance(path, str) and bool(path), path, 'nonempty class path')
                        check(f'{branch} probability', finite(probability) and 0 <= probability <= 1, probability, 'finite value in [0, 1]')
                if index == 1:
                    baseline = candidates
                if index == 2:
                    check('Classification disabled', classification is None, classification, None)
                    check('Baseline prediction available', baseline is not None, baseline is not None, True)
                    if baseline is not None:
                        check('Same retrieval count', len(candidates) == len(baseline), len(candidates), len(baseline))
                        for n, (before, after) in enumerate(zip(baseline, candidates), 1):
                            check(f'Candidate {n} same class', before.get('class_path') == after.get('class_path'), after.get('class_path'), before.get('class_path'))
                            a, b = before.get('similarity'), after.get('similarity')
                            check(f'Candidate {n} same similarity', finite(a) and finite(b) and abs(a-b) <= 1e-5, b, f'{a} +/- 0.00001')
            elif index == 4:
                detail = data.get('detail')
                check('Invalid-image error detail', isinstance(detail, str) and 'Invalid image_base64' in detail, detail, 'contains Invalid image_base64')
            else:
                field = 'topk' if index == 5 else 'file'
                detail = data.get('detail')
                valid = isinstance(detail, list) and any(field in error.get('loc', []) for error in detail if isinstance(error, dict))
                check('Validation error field', valid, detail, field)

    failed = sum(not all(c['passed'] for c in r['checks']) for r in results)
    checks = [c for r in results for c in r['checks']]
    summary = {'tests': len(results), 'passed': len(results)-failed, 'failed': failed,
               'checks_passed': sum(c['passed'] for c in checks), 'checks_total': len(checks)}
    display('\nSummary: ' + json.dumps(summary))
    if options.report:
        options.report.write_text(redact(json.dumps({'summary': summary, 'tests': results}, indent=2, ensure_ascii=False)) + '\n')
        display(f'Detailed report saved to {options.report}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
