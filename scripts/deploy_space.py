"""Publish the site (site/) as a static Hugging Face Space.

    python scripts/deploy_space.py --repo-id <hf-username>/judge-calibration --dry-run
    python scripts/deploy_space.py --repo-id <hf-username>/judge-calibration

Needs `pip install -e ".[site]"` and a one-time `hf auth login` with a *write*
token. This publishes - run it only when you mean to.

The Space gets exactly what GitHub Pages serves: site/ as-is, plus a README.md
carrying the Space's settings (static SDK, title, licence) and the credits. The
upload replaces everything in the Space in one commit, so files removed from
site/ disappear from the Space too. The page's canonical URL stays the GitHub
Pages address, so search engines credit the original.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SITE = Path(__file__).resolve().parents[1] / "site"
PAGES_URL = "https://senguptashubham.github.io/judge-calibration/"
REPO_URL = "https://github.com/senguptashubham/judge-calibration"

SPACE_CARD = f"""---
title: Do LLM Judges Know When They're Wrong?
emoji: ⚖️
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Do LLM judges know when they're wrong? Try to trick them.
tags:
  - llm-as-a-judge
  - evaluation
  - calibration
  - mt-bench
---

# Do LLM Judges Know When They're Wrong?

An interactive tour of a study of three open-weight LLM judges (Qwen2.5-7B-Instruct,
kev-8b, auto-j-13b) graded against human votes on 1,836 MT-Bench comparisons:

- try to trick a judge by swapping or padding the answers;
- play against the judges;
- compare the three;
- see how many wrong verdicts an auto-accept pipeline would let through.

Every verdict shown is what the judges actually produced in the recorded runs.
Nothing is generated live.

- **Also at:** {PAGES_URL}
- **Code, full report and data build:** {REPO_URL}

**Author:** Shubham Sengupta ([GitHub](https://github.com/senguptashubham),
[LinkedIn](https://www.linkedin.com/in/senguptashubham/)).

**Licences:**
- **Code:** MIT.
- **Text and figures:** CC BY 4.0.
- **Data:** [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments),
  CC BY 4.0 (Zheng et al., NeurIPS 2023).
- **Fonts:** Inter and JetBrains Mono, SIL OFL 1.1 (licence texts in `fonts/`).
"""


def stage(target: Path) -> list[str]:
    """Copy site/ and the Space card into `target`; return the staged paths."""
    shutil.copytree(SITE, target, dirs_exist_ok=True)
    (target / "README.md").write_text(SPACE_CARD, encoding="utf-8")
    return sorted(str(p.relative_to(target)).replace("\\", "/") for p in target.rglob("*") if p.is_file())


def main(repo_id: str, dry_run: bool) -> None:
    if not (SITE / "data" / "site-data.js").exists():
        raise SystemExit("site/data/ is not built - run `python -m analysis.site_data --config configs/run.yaml` first")
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        print(f"{len(files)} files for spaces/{repo_id}:")
        for f in files:
            print("  ", f)
        if dry_run:
            print("Dry run: nothing uploaded.")
            return
        api = HfApi()
        api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          commit_message="Publish the site", delete_patterns="*")
    print(f"Published: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="<hf-username>/<space-name>")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    args = parser.parse_args()
    main(args.repo_id, args.dry_run)
