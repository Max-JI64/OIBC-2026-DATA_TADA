"""Read-only document, saved-metadata and public tracked-file audit. No models."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/2026-10-22_project_history_review"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    documents = [ROOT / name for name in ("README.md", "LOG.md", "AGENTS.md", "AGENT_MEMORY.md", "데이터_확인_결과.md")]
    documents += sorted((ROOT / "reports").rglob("*.md"))
    missing, links, report_rows, symbols = [], 0, [], []
    for path in documents:
        if not path.exists():
            missing.append(dict(source="inventory", target=str(path.relative_to(ROOT))))
            continue
        body = path.read_text(encoding="utf-8-sig")
        if chr(0x00B7) in body:
            symbols.append(str(path.relative_to(ROOT)))
        for m in re.finditer(r"\]\(([^)]+)\)", body):
            target = m.group(1).strip("<>")
            if re.match(r"^[A-Za-z]+://", target) or target.startswith("#"):
                continue
            location = unquote(target.split("#", 1)[0])
            file = (path.parent / location).resolve()
            links += 1
            if not file.exists():
                missing.append(dict(source=str(path.relative_to(ROOT)), target=target))
        if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}_", path.name):
            first = body.splitlines()
            report_rows.append(dict(path=str(path.relative_to(ROOT)), title=first[0],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                logged=path.name in (ROOT / "reports/logs" / f"{path.name[:10]}.md").read_text(encoding="utf-8")))
    metadata = []
    # Read aggregate JSON records only, never raw/prediction tables or model files.
    for parent in ("analysis", "eda", "validation", "final"):
        for path in sorted((ROOT / "data" / parent).rglob("*.json")):
            if path.name not in ("summary.json", "verification.json") or "executed_code" in path.parts:
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                continue
            metadata.append(dict(path=str(path.relative_to(ROOT)), status=obj.get("status"), selection=obj.get("selection", obj.get("chosen_method")),
                fits=obj.get("model_fit_count", obj.get("new_fits")), V2_executed=obj.get("V2_executed"),
                official_submission_executed=obj.get("official_submission_executed")))
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8").split("\0")
    forbidden = [p for p in tracked if p and (p.lower().endswith(".csv") or p.lower().startswith(("data/", "oibc_2026_data/")))]
    unlogged = [r["path"] for r in report_rows if not r["logged"]]
    result = dict(status="passed" if not missing and not forbidden and not unlogged else "attention",
        at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes"), documents=len(documents), local_links=links,
        reports=report_rows, missing_links=missing, unlogged_reports=unlogged, forbidden_tracked=forbidden,
        historical_middle_dot_files=symbols, saved_metadata=metadata, model_calls=0, raw_table_reads=0,
        scope="Document/metadata audit only; prior numerical analyses, plots and tests not rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "documents", "local_links", "missing_links", "unlogged_reports", "forbidden_tracked", "historical_middle_dot_files")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
