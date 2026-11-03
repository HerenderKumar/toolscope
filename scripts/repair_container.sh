#!/usr/bin/env bash
# Apply every environment repair (D-024, D-027, D-028, D-030) to a LiveMCPBench container. Idempotent.
# Run after the servers have been started once (so uv/npx caches exist), e.g. after a tool check.
# Usage: bash scripts/repair_container.sh <container>
set -euo pipefail
CTR="${1:-LiveMCPBench_container}"

docker exec -i "$CTR" bash -s <<'INNER'
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
chmod 1777 /tmp                                                     # D-024: /tmp must be world-writable
UA_SUFFIX="ToolScope research"

# D-035: the benchmark ships the git task repository as annotated_data/git.bundle (a repo cannot be committed
# inside a repo); /root/git links to annotated_data/git, which must be unpacked from the bundle
if [ -f /LiveMCPBench/annotated_data/git.bundle ] && [ ! -d /LiveMCPBench/annotated_data/git ]; then
  git clone -q /LiveMCPBench/annotated_data/git.bundle /LiveMCPBench/annotated_data/git
fi

# D-024: Python 3.12 with SQLite FTS5 for basic-memory
if [ ! -x /usr/bin/python3.12 ]; then
  apt-get update -qq && apt-get install -y -qq software-properties-common >/dev/null \
    && add-apt-repository -y ppa:deadsnakes/ppa >/dev/null && apt-get update -qq \
    && apt-get install -y -qq python3.12 python3.12-venv >/dev/null
fi
# D-027: virtual display for the headed Playwright browser
command -v xvfb-run >/dev/null || apt-get install -y -qq xvfb xauth >/dev/null

# D-028: OSM server sends a User-Agent to Overpass
for f in /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/osm_mcp_server/server.py; do
  [ -f "$f" ] && sed -i "s/aiohttp.ClientSession()/aiohttp.ClientSession(headers={\"User-Agent\": \"OSM-MCP-Server\/1.0 ($UA_SUFFIX)\"})/g" "$f"
done

# D-042: OSM search_category with subcategories built invalid Overpass QL ('[("amenity"="a" or "amenity"="b")]',
# always HTTP 400); use Overpass's regex form '["amenity"~"^(a|b)$"]', which is what the tool documents
for f in /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/osm_mcp_server/server.py; do
  [ -f "$f" ] && python3 - "$f" <<'PYFIX'
import sys
p = sys.argv[1]; s = open(p).read()
old = """            subcategory_filters = " or ".join([f'"{category}"="{sub}"' for sub in subcategories])
            query_filter = f'({subcategory_filters})'"""
new = """            query_filter = f'"{category}"~"^(' + "|".join(subcategories) + ')$"'  # ToolScope D-042"""
if old in s:
    open(p, "w").write(s.replace(old, new))
PYFIX
done

# D-050: yahoo-finance get_earning_dates advertises `limit` as a string but passes it on unconverted to yfinance,
# which compares it with ints ("'<' not supported between 'int' and 'str'"); an int is rejected by the schema
for f in /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/mcp_yahoo_finance/server.py; do
  [ -f "$f" ] && sed -i 's|stock.get_earnings_dates(limit=limit)|stock.get_earnings_dates(limit=int(limit))  # ToolScope D-050|' "$f"
done

# D-052: puppeteer runs as root in the container, where Chromium refuses to start without --no-sandbox
# ("Running as root without --no-sandbox is not supported"); the server has a Docker mode (headless, --no-sandbox,
# --single-process, --no-zygote) switched on by DOCKER_CONTAINER, which the catalog config does not set
for f in /root/.npm/_npx/*/node_modules/@modelcontextprotocol/server-puppeteer/dist/index.js; do
  [ -f "$f" ] && sed -i 's|deepMerge(process.env.DOCKER_CONTAINER ? docker_args|deepMerge(true /* ToolScope D-052 */ ? docker_args|' "$f"
done

# D-055: mermaid-validator passes the diagram to mermaid-cli as `-i /dev/stdin`; Node connects a child's stdin
# through a socket on Linux, which cannot be opened as a file (ENXIO), so every diagram is "invalid". Write it to a
# temp file instead, and give mermaid-cli's headless Chrome --no-sandbox (the container runs as root, cf. D-052)
echo '{"args": ["--no-sandbox", "--disable-setuid-sandbox"]}' > /root/.mmdc-puppeteer.json
for f in /root/.npm/_npx/*/node_modules/@rtuin/mcp-mermaid-validator/dist/main.js; do
  [ -f "$f" ] && python3 - "$f" <<'PYFIX'
import sys
p = sys.argv[1]; s = open(p).read()
if "ToolScope D-055" not in s:
    s = s.replace('import { spawn } from "child_process";',
                  'import { spawn } from "child_process";\nimport { writeFileSync } from "fs";  // ToolScope D-055')
    s = s.replace('            const args = [\n                "@mermaid-js/mermaid-cli",\n                "-i",\n                "/dev/stdin",',
                  '            const tsInput = `/tmp/mermaid_${process.pid}_${Date.now()}.mmd`;  // ToolScope D-055\n'
                  '            writeFileSync(tsInput, diagram);\n'
                  '            const args = [\n                "@mermaid-js/mermaid-cli",\n                "-p",\n'
                  '                "/root/.mmdc-puppeteer.json",\n                "-i",\n                tsInput,')
    open(p, "w").write(s)
PYFIX
done

# D-056: youtube-transcript 0.1.1 (newest) returns {toolResult: {content, isError}} (pre-2024 MCP shape); current MCP
# clients require top-level `content` and reject the result ("content: Field required") even when the transcript
# was fetched. Spread the inner object into the result instead.
for f in /root/.npm/_npx/*/node_modules/@kimtaeyoon83/mcp-server-youtube-transcript/dist/index.js; do
  [ -f "$f" ] && grep -q "toolResult: {" "$f" && sed -i 's|toolResult: {|...{ /* ToolScope D-056 */|' "$f"
done

mkdir -p /root/Documents   # D-047: taskmanager stores ~/Documents/tasks.json and fails (ENOENT) without the folder

# D-045: ppt server (office-powerpoint-mcp-server 2.0.7, also the newest): create/open never set the "current"
# presentation (presentation_tools.py: "this would need to be handled by caller"; the wrappers are never applied),
# so every tool documented as "uses current if None" fails. Fall back to the most recently stored presentation.
for f in /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/ppt_mcp_server.py; do
  [ -f "$f" ] && python3 - "$f" <<'PYFIX'
import sys
p = sys.argv[1]; s = open(p).read()
old = '''    """Get the current presentation ID."""
    return current_presentation_id'''
new = '''    """Get the current presentation ID."""
    if current_presentation_id in presentations:  # ToolScope D-045
        return current_presentation_id
    return next(reversed(presentations), None)'''
if old in s:
    open(p, "w").write(s.replace(old, new))
PYFIX
done

# D-030: Wikipedia server: Wikimedia blocks any User-Agent containing "WikipediaMCPServer" (measured:
# JSONDecodeError on an HTML block page), and the catalog-era 0.1.0 sends no User-Agent on its raw requests.
for f in /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/wikipedia_mcp/wikipedia_client.py; do
  [ -f "$f" ] || continue
  sed -i "s|self.user_agent = .*|self.user_agent = \"ToolScope/1.0 (research benchmark) wikipedia-mcp\"|" "$f"
  sed -i "s|params=params)$|params=params, headers={\"User-Agent\": self.user_agent})|" "$f"
  rm -rf "$(dirname "$f")/__pycache__"
done

# D-027/D-030: browser build matching the Playwright MCP server's own playwright-core
source /root/.nvm/nvm.sh >/dev/null 2>&1
for d in /root/.npm/_npx/*/; do
  if [ -d "$d/node_modules/@executeautomation/playwright-mcp-server" ]; then
    (cd "$d/node_modules/playwright-core" && node cli.js install chromium 2>&1 | grep -E "downloaded" || true)
  fi
done

echo "osm_patched=$(grep -l "$UA_SUFFIX" /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/osm_mcp_server/server.py 2>/dev/null | wc -l)" \
     "wikipedia_patched=$(grep -l "ToolScope/1.0 (research benchmark)" /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/wikipedia_mcp/wikipedia_client.py 2>/dev/null | wc -l)" \
     "browsers=$(ls /root/.cache/ms-playwright | tr '\n' ',')" \
     "osm_subcat_fix=$(grep -l "ToolScope D-042" /root/.cache/uv/archive-v0/*/lib/python3.*/site-packages/osm_mcp_server/server.py 2>/dev/null | wc -l) git_repo=$([ -d /root/git/.git ] && echo yes || echo no) python3.12=$([ -x /usr/bin/python3.12 ] && echo yes || echo no) xvfb=$(command -v xvfb-run >/dev/null && echo yes || echo no) tmp=$(stat -c %a /tmp)"
INNER
