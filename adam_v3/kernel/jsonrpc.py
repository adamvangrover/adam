import json

class JsonRpcHandler:
    def handle(self, request):
        if isinstance(request, str):
            request = json.loads(request)

        req_id = request.get("id", None)
        method = request.get("method")
        params = request.get("params", {})

        if request.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": req_id}

        # Handle the request routing and execution (mocked logic here for kernel handling)
        result_payload = {"status": "success", "executed_method": method}

        return {
            "jsonrpc": "2.0",
            "result": result_payload,
            "id": req_id
        }
