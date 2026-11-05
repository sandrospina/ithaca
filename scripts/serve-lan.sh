#!/usr/bin/env bash
# Builds the release web client and serves it to the local network, so other
# computers on the same network can test it without a push.
#
#   scripts/serve-lan.sh            # build, then serve on port 8081
#   scripts/serve-lan.sh --port 9000
#   scripts/serve-lan.sh --no-build # serve the last build again
#
# Plain http on a network address is not a secure context: fine for the current
# build, but WebGPU will need the GitHub Pages site (https) or localhost.
set -euo pipefail

port=8081
build=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --no-build) build=0; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "$0")/.." && pwd)"
site="$root/target/lan-site"

if [[ $build == 1 ]]; then
  echo "building the release web client into target/lan-site…"
  (cd "$root/crates/rl-web" && trunk build --release --cargo-profile wasm-release --dist "$site" --public-url /)
fi
[[ -f "$site/index.html" ]] || { echo "no build in $site: run without --no-build" >&2; exit 1; }

echo
echo "serving on port $port. Open one of these on the other computer:"
for iface in en0 en1 en2; do
  ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
  [[ -n "$ip" ]] && echo "  http://$ip:$port/"
done
echo "  (this computer: http://localhost:$port/)"
echo "macOS may ask once whether Python may accept incoming connections: allow it."
echo "Ctrl-C stops the server."
echo
exec python3 -m http.server "$port" --bind 0.0.0.0 --directory "$site"
