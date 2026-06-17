import json, hashlib
payload = {"status": "ok", "value": 42}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest())
