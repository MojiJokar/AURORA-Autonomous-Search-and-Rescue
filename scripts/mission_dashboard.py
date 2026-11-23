"""Serve the AURORA mission dashboard with HTTP video range support."""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 256 * 1024


def make_handler(report_path: Path, video_path: Path, telemetry_path=None):
    dashboard_path = PROJECT_ROOT / "dashboard" / "index.html"

    class DashboardHandler(BaseHTTPRequestHandler):

        def send_bytes(self, content, content_type, status=200, head_only=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            if not head_only:
                self.wfile.write(content)

        def serve_video(self, head_only=False):
            if not video_path.is_file():
                self.send_error(404, "Annotated video not found")
                return

            file_size = video_path.stat().st_size
            if file_size == 0:
                self.send_error(404, "Annotated video is empty")
                return

            range_header = self.headers.get("Range")
            start = 0
            end = file_size - 1
            status = 200

            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())

                if not match or (not match.group(1) and not match.group(2)):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                first, last = match.groups()

                if first:
                    start = int(first)
                    end = int(last) if last else file_size - 1
                else:
                    suffix_length = int(last)
                    if suffix_length <= 0:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    start = max(0, file_size - suffix_length)

                if start >= file_size or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                end = min(end, file_size - 1)
                status = 206

            content_length = end - start + 1

            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store")

            if status == 206:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{end}/{file_size}",
                )

            self.end_headers()

            if head_only:
                return

            try:
                with video_path.open("rb") as video_file:
                    video_file.seek(start)
                    remaining = content_length

                    while remaining > 0:
                        chunk = video_file.read(min(CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

            except (BrokenPipeError, ConnectionResetError):
                pass

        def handle_request(self, head_only=False):
            route = urlparse(self.path).path

            if route == "/":
                if not dashboard_path.is_file():
                    self.send_error(404, "Dashboard HTML file not found")
                    return

                self.send_bytes(
                    dashboard_path.read_bytes(),
                    "text/html; charset=utf-8",
                    head_only=head_only,
                )
                return

            if route == "/api/report":
                if not report_path.is_file():
                    content = json.dumps({
                        "error": "Mission report not found",
                        "expected_path": str(report_path),
                    }).encode("utf-8")

                    self.send_bytes(
                        content,
                        "application/json",
                        status=404,
                        head_only=head_only,
                    )
                    return

                try:
                    content = report_path.read_bytes()
                    json.loads(content)
                except (OSError, json.JSONDecodeError):
                    self.send_error(500, "Mission report is not valid JSON")
                    return

                self.send_bytes(
                    content,
                    "application/json",
                    head_only=head_only,
                )
                return

            if route == "/api/telemetry":
                if telemetry_path is None or not telemetry_path.is_file():
                    content = json.dumps({
                        "available": False,
                        "message": "No telemetry snapshot is available",
                    }).encode("utf-8")

                    self.send_bytes(
                        content,
                        "application/json",
                        head_only=head_only,
                    )
                    return

                try:
                    stat = telemetry_path.stat()
                    content = telemetry_path.read_bytes()
                    telemetry = json.loads(content)

                    if not isinstance(telemetry, dict):
                        raise ValueError("Telemetry JSON must be an object")

                    from aurora.telemetry.health import (
                        classify_telemetry_source,
                    )
                    from datetime import datetime, timezone
                    import time

                    age_seconds = max(0.0, time.time() - stat.st_mtime)
                    source = telemetry.get("source", "unknown")

                    health_status = classify_telemetry_source(
                        source,
                        age_seconds,
                    )

                    telemetry["health"] = {
                        "status": health_status,
                        "age_seconds": round(age_seconds, 2),
                        "stale_after_seconds": 3.0,
                        "disconnect_after_seconds": 10.0,
                        "last_update": datetime.fromtimestamp(
                            stat.st_mtime,
                            timezone.utc,
                        ).isoformat(),
                        "is_live": health_status == "LIVE",
                    }

                    content = json.dumps(telemetry).encode("utf-8")

                except (OSError, json.JSONDecodeError, ValueError):
                    self.send_error(
                        500,
                        "Telemetry snapshot is not valid JSON",
                    )
                    return

                self.send_bytes(
                    content,
                    "application/json",
                    head_only=head_only,
                )
                return

            if route == "/video":
                self.serve_video(head_only=head_only)
                return

            self.send_error(404, "Not found")

        def do_GET(self):
            self.handle_request(head_only=False)

        def do_HEAD(self):
            self.handle_request(head_only=True)

        def log_message(self, format_string, *args):
            print("Dashboard:", format_string % args)

    return DashboardHandler


def main():
    parser = argparse.ArgumentParser(
        description="Start the local AURORA mission dashboard."
    )
    parser.add_argument(
        "--report",
        default="datasets/aurora_pipeline_output.json",
        help="Mission JSON report",
    )
    parser.add_argument(
        "--video",
        default="datasets/aurora_pipeline_output.mp4",
        help="Annotated MP4 video",
    )
    parser.add_argument(
        "--telemetry",
        default="datasets/telemetry_latest.json",
        help="Latest telemetry snapshot JSON",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    report_path = Path(args.report).resolve()
    video_path = Path(args.video).resolve()
    telemetry_path = Path(args.telemetry).resolve()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(report_path, video_path, telemetry_path),
    )

    print("AURORA Mission Dashboard")
    print("------------------------")
    print(f"Open in your browser: http://{args.host}:{args.port}")
    print("Video byte-range support: enabled")
    print("Press Ctrl+C in this terminal to stop the dashboard.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AURORA dashboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
