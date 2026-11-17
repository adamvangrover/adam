"""
Zero-Dependency FastAPI Compatibility Shim & Stdlib HTTP Server
=================================================================
Allows AFOS IDE & Command Center servers to import and serve routes in standard
Python 3.11+ environments even when `fastapi` is not installed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ValidationError


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(str(detail or status_code))


class _StatusCodes:
    HTTP_200_OK = 200
    HTTP_400_BAD_REQUEST = 400
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404
    HTTP_413_REQUEST_ENTITY_TOO_LARGE = 413
    HTTP_422_UNPROCESSABLE_ENTITY = 422
    HTTP_500_INTERNAL_SERVER_ERROR = 500


status = _StatusCodes()


class HTMLResponse:
    def __init__(self, content: str = "", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code


class CORSMiddleware:
    pass


class _RouteEntry:
    def __init__(self, method: str, path: str, func: Callable[..., Any], response_class: Any = None) -> None:
        self.method = method.upper()
        self.path = path
        self.func = func
        self.response_class = response_class


class FastAPI:
    """Minimal FastAPI-compatible router & stdlib HTTP runner fallback."""

    def __init__(
        self,
        title: str = "AFOS API",
        description: str = "",
        version: str = "30.1.0",
        docs_url: Optional[str] = None,
        redoc_url: Optional[str] = None,
    ) -> None:
        self.title = title
        self.description = description
        self.version = version
        self.docs_url = docs_url
        self.redoc_url = redoc_url
        self._routes: List[_RouteEntry] = []

    def add_middleware(self, middleware_cls: Any, **kwargs: Any) -> None:
        return None

    def get(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        response_class = kwargs.get("response_class")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append(_RouteEntry("GET", path, func, response_class=response_class))
            return func

        return decorator

    def post(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        response_class = kwargs.get("response_class")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append(_RouteEntry("POST", path, func, response_class=response_class))
            return func

        return decorator

    def _match_route(self, method: str, req_path: str) -> Tuple[Optional[_RouteEntry], Dict[str, str]]:
        method = method.upper()
        for entry in self._routes:
            if entry.method != method:
                continue
            if entry.path == req_path:
                return entry, {}
            if "{" in entry.path:
                e_parts = entry.path.strip("/").split("/")
                r_parts = req_path.strip("/").split("/")
                if len(e_parts) == len(r_parts):
                    params: Dict[str, str] = {}
                    matched = True
                    for ep, rp in zip(e_parts, r_parts):
                        if ep.startswith("{") and ep.endswith("}"):
                            params[ep[1:-1]] = rp
                        elif ep != rp:
                            matched = False
                            break
                    if matched:
                        return entry, params
        return None, {}

    def dispatch_request(
        self,
        method: str,
        raw_path: str,
        body_bytes: bytes = b"",
    ) -> Tuple[int, str, bytes]:
        parsed = urlparse(raw_path)
        entry, path_params = self._match_route(method, parsed.path)
        if not entry:
            payload = json.dumps({"detail": f"Route not found: {parsed.path}"}).encode("utf-8")
            return 404, "application/json; charset=utf-8", payload

        query_raw = parse_qs(parsed.query)
        query_params = {k: v[-1] for k, v in query_raw.items() if v}

        body_data: Dict[str, Any] = {}
        if body_bytes:
            try:
                body_data = json.loads(body_bytes.decode("utf-8"))
            except Exception as exc:
                payload = json.dumps({"detail": f"Invalid JSON body: {exc}"}).encode("utf-8")
                return 422, "application/json; charset=utf-8", payload

        sig = inspect.signature(entry.func)
        call_kwargs: Dict[str, Any] = {}
        try:
            # Resolve annotations safely for postponed annotations (`from __future__ import annotations`)
            func_globals = getattr(entry.func, "__globals__", {})
            builtin_types = {"str": str, "int": int, "float": float, "bool": bool}
            for name, param in sig.parameters.items():
                ann = param.annotation
                if isinstance(ann, str):
                    ann = builtin_types.get(ann) or func_globals.get(ann, Any)
                if name in path_params:
                    call_kwargs[name] = path_params[name]
                elif isinstance(ann, type) and issubclass(ann, BaseModel):
                    call_kwargs[name] = ann(**body_data)
                elif name in query_params:
                    raw_val = query_params[name]
                    if ann is int:
                        call_kwargs[name] = int(raw_val)
                    elif ann is float:
                        call_kwargs[name] = float(raw_val)
                    elif ann is bool:
                        call_kwargs[name] = raw_val.lower() in ("1", "true", "yes")
                    else:
                        call_kwargs[name] = raw_val
                elif name in body_data:
                    call_kwargs[name] = body_data[name]
                elif param.default is not inspect.Parameter.empty:
                    call_kwargs[name] = param.default
                else:
                    raise HTTPException(status_code=422, detail=f"Missing required parameter: {name}")

            if inspect.iscoroutinefunction(entry.func):
                result = asyncio.run(entry.func(**call_kwargs))
            else:
                result = entry.func(**call_kwargs)
        except ValidationError as exc:
            return 422, "application/json; charset=utf-8", json.dumps({"detail": exc.errors()}, default=str).encode("utf-8")
        except HTTPException as exc:
            return exc.status_code, "application/json; charset=utf-8", json.dumps({"detail": exc.detail}).encode("utf-8")
        except Exception as exc:
            return 500, "application/json; charset=utf-8", json.dumps({"detail": str(exc)}).encode("utf-8")

        if isinstance(result, HTMLResponse):
            return result.status_code, "text/html; charset=utf-8", result.content.encode("utf-8")
        if entry.response_class is HTMLResponse and isinstance(result, str):
            return 200, "text/html; charset=utf-8", result.encode("utf-8")
        if isinstance(result, BaseModel):
            return 200, "application/json; charset=utf-8", result.model_dump_json().encode("utf-8")
        return 200, "application/json; charset=utf-8", json.dumps(result, default=str).encode("utf-8")

    def serve_stdlib(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        fastapi_app = self

        class _Handler(BaseHTTPRequestHandler):
            def _handle(self, method: str) -> None:
                content_len = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(content_len) if content_len > 0 else b""
                code, ctype, resp_bytes = fastapi_app.dispatch_request(method, self.path, body)
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "SAMEORIGIN")
                self.end_headers()
                self.wfile.write(resp_bytes)

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return None

        server = ThreadingHTTPServer((host, port), _Handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
