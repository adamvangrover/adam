import json

class JsonRpcHandler:
    def handle(self, request):
        return {"jsonrpc": "2.0", "result": "handled", "id": 1}
