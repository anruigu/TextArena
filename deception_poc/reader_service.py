#!/usr/bin/env python3
"""HTTP leakage service: loads the frozen leak-READER (Qwen3-8B) + the fitted linear probe ONCE and
serves P(strong | public channel) to all self-play rollout workers, so the 8B is not duplicated per
worker. The outer arms-race loop refits the probe (train_reader_probe.py) and POSTs /reload to
hot-swap the .npz between rounds without restarting the model.

Endpoints (JSON):
  POST /leakage  {"public": ["<public channel text>", ...]}   -> {"p_strong": [..]}
  POST /reload   {"probe": "/path/to/probe_leakreader_<tag>.npz"} -> {"ok": true, "layer": L}
  GET  /health   -> {"ok": true, "probe": "...", "layer": L}

  cd /workspace/allie/TextArena/deception_poc
  CUDA_VISIBLE_DEVICES=0 /workspace/allie/performative/.venv/bin/python reader_service.py \
      --probe probes/probe_leakreader_qwen3_8b.npz --port 8137
  # then: export READER_BASE_URL=http://127.0.0.1:8137
"""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from leaky_reward import LeakReader

_LOCK = threading.Lock()
STATE = {"reader": None, "probe": None}


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                r = STATE["reader"]
                self._send(200, {"ok": r is not None, "probe": STATE["probe"],
                                 "layer": (r.layer if r else None)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._send(400, {"error": "bad json"}); return
            if self.path == "/leakage":
                pub = data.get("public", [])
                if isinstance(pub, str):
                    pub = [pub]
                with _LOCK:
                    ps = STATE["reader"].leakage(pub, already_public=True)
                self._send(200, {"p_strong": [float(x) for x in ps]})
            elif self.path == "/reload":
                probe = data["probe"]
                with _LOCK:
                    # keep the loaded model; only re-read the linear probe weights
                    import numpy as np
                    d = np.load(probe, allow_pickle=True)
                    r = STATE["reader"]
                    r.w = d["w"].astype(np.float32)
                    r.b = float(d["b"][0]) if np.ndim(d["b"]) else float(d["b"])
                    r.mu = d["mu"].astype(np.float32)
                    r.sd = d["sd"].astype(np.float32)
                    r.layer = int(d["layer"])
                    STATE["probe"] = probe
                self._send(200, {"ok": True, "layer": STATE["reader"].layer})
            else:
                self._send(404, {"error": "not found"})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="probes/probe_leakreader_qwen3_8b.npz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8137)
    args = ap.parse_args()
    STATE["reader"] = LeakReader(probe_path=args.probe, device=args.device, batch=16)
    STATE["probe"] = args.probe
    print(f"[reader_service] loaded {args.probe} (layer {STATE['reader'].layer}) on {args.device}; "
          f"serving http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler()).serve_forever()


if __name__ == "__main__":
    main()
