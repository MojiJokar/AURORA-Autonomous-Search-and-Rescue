"""Serve the AURORA mission dashboard on the local computer."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_handler(report_path: Path, video_path: Path):
    dashboard_path = PROJECT_ROOT / "dashboard" / "index.html"

    class DashboardHandler(BaseHTTPRequestHandler):
        def send_bytes(self, content, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            route = urlparse(self.path).path

            if route == "/":
                if not dashboard_path.is_file():
                    self.send_error(404, "Dashboard HTML file not found")
                    return
                self.send_bytes(
                    dashboard_path.read_bytes(),
                    "text/html; charset=utf-8",
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
                    )
                    return

                try:
                    content = report_path.read_bytes()
                    json.loads(content)
                except (OSError, json.JSONDecodeError):
                    self.send_error(500, "Mission report is not valid JSON")
                    return

                self.send_bytes(content, "application/json")
                return

            if route == "/video":
                if not video_path.is_file():
                    self.send_error(404, "Annotated video not found")
                    return
                self.send_bytes(video_path.read_bytes(), "video/mp4")
                return

            self.send_error(404, "Not found")

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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    report_path = Path(args.report).resolve()
    video_path = Path(args.video).resolve()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(report_path, video_path),
    )

    print("AURORA Mission Dashboard")
    print("------------------------")
    print(f"Open in your browser: http://{args.host}:{args.port}")
    print("Press Ctrl+C in this terminal to stop the dashboard.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AURORA dashboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
