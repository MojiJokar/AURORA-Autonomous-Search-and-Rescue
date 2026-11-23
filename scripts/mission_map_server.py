"""Serve the AURORA geospatial mission map and GeoJSON."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--geojson",
        default="datasets/geolocated_targets.geojson",
    )
    args = parser.parse_args()
    geojson_path = (ROOT / args.geojson).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                path = ROOT / "dashboard" / "map.html"
                content_type = "text/html; charset=utf-8"
            elif self.path == "/api/targets":
                path = geojson_path
                content_type = "application/geo+json"
            else:
                self.send_error(404)
                return

            if not path.is_file():
                self.send_error(404, "Map data not found. Generate GeoJSON first.")
                return

            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, fmt, *args):
            print("Mission map:", fmt % args)

    print(f"AURORA mission map: http://{args.host}:{args.port}")
    print(f"GeoJSON file: {geojson_path}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
