"""Bound multipart body before Starlette parses or spools it."""
from fastapi.responses import JSONResponse


class UploadLimit:
    def __init__(self, app):
        self.app = app
        self.limit = 10 * 1024 * 1024 + 64 * 1024

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/v1/eval/predict":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.limit:
            return await JSONResponse({"detail": "Request body exceeds limit"}, 413)(
                scope, receive, send,
            )
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.limit:
                return await JSONResponse({"detail": "Request body exceeds limit"}, 413)(
                    scope, receive, send,
                )
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        payload = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}

        await self.app(scope, replay, send)
