"""Run the local SVM web demo: python run_web.py [--port 5000]."""
from __future__ import annotations

import argparse

from webapp import create_app


def main():
    parser = argparse.ArgumentParser(description="Web kiểm thử SVM trên máy cá nhân")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app = create_app()
    engine = app.extensions["analysis_engine"]
    if engine.default:
        engine.load(engine.default)
    print(f"\nSVM News Lab: http://127.0.0.1:{args.port}\n", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
