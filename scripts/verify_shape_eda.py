"""Check the local EDA artifacts and make half-size contact sheets for review."""
from html.parser import HTMLParser
import json
from pathlib import Path
import py_compile
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eda" / "2026-10-19_basic_shape"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(value for key, value in attrs if key in ("src", "href"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for name in ["basic_shape_eda.py", "inspect_time_precision.py", "verify_shape_eda.py"]:
        py_compile.compile(str(ROOT / "scripts" / name), doraise=True)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    gallery = json.loads((OUT / "gallery.json").read_text(encoding="utf-8"))
    assert summary["source_unchanged"] is True
    assert len(summary["files"]) == 10
    assert len(gallery) == summary["figure_count"]
    numeric_count = 0
    for rel, f in summary["files"].items():
        assert f["key_null_rows"] == 0, rel
        for col, p in f["numeric"].items():
            assert p["count"] + p["nulls"] + p["infinite"] == f["rows"], (rel, col)
            assert p["negative"] + p["zeros"] <= p["count"], (rel, col)
            q = [p[key] for key in ["min", "q001", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q999", "max"]]
            assert np.isfinite(q).all() and np.all(np.diff(q) >= 0), (rel, col)
            assert len(p["ecdf_x"]) == len(p["ecdf_y"]) == 501
            assert np.all(np.diff(p["ecdf_x"]) >= 0), (rel, col)
            numeric_count += 1
    parser = Links()
    parser.feed((OUT / "index.html").read_text(encoding="utf-8"))
    for path in parser.paths:
        assert (OUT / path).is_file(), path
    for g in gallery:
        with Image.open(OUT / g["path"]) as im:
            im.verify()
    qa = OUT / "qa"
    qa.mkdir(exist_ok=True)
    for start in range(0, len(gallery), 4):
        group = gallery[start:start + 4]
        resized = []
        for g in group:
            with Image.open(OUT / g["path"]) as source:
                im = source.convert("RGB")
                im = im.resize((im.width // 2, im.height // 2), Image.Resampling.LANCZOS)
                resized.append((g, im))
        width = max(im.width for _, im in resized)
        height = max(im.height for _, im in resized) + 26
        sheet = Image.new("RGB", (width * 2, height * 2), "#ededed")
        draw = ImageDraw.Draw(sheet)
        for i, (g, im) in enumerate(resized):
            x, y = (i % 2) * width, (i // 2) * height
            sheet.paste(im, (x, y + 26))
            draw.text((x + 5, y + 5), Path(g["path"]).stem, fill="black")
        sheet.save(qa / f"contact_{start // 4 + 1:02d}.png")
    result = {"input_files": 10, "numeric_profiles_checked": numeric_count,
              "figures_checked": len(gallery), "html_links_checked": len(parser.paths),
              "source_unchanged": summary["source_unchanged"],
              "contact_sheets": (len(gallery) + 3) // 4,
              "python_files_compiled": 3,
              "note": "Artifact checks passed. Human visual review is a separate step."}
    (qa / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
