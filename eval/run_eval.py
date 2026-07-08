#!/usr/bin/env python3
"""Évaluation non-régression du RAG : recall@5, MRR du retrieval + présence des mots-clés.

Jeu d'or : `eval/golden.jsonl`, un objet JSON par ligne :

    {"question": "...", "collection": "ma_coll", "expected_doc": "fichier.pdf",
     "expected_keywords": ["mot", "clé"]}

- `expected_doc` : nom de source attendu parmi les sources retrouvées (recall / MRR).
- `expected_keywords` : mots/expressions attendus dans la réponse générée (insensible à la
  casse). Désactivable avec `--no-generate` (métriques de retrieval uniquement).

Usage :
    python -m rag_builder.… ? Non — script autonome :
    python eval/run_eval.py --golden eval/golden.jsonl [--no-generate] [--output eval/report.json]

⚠️ En mode Qdrant local, arrêter l'API avant (accès exclusif au dossier storage/qdrant).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_builder.config import get_settings  # noqa: E402
from rag_builder.core.rag_service import RagService  # noqa: E402


def _build_service(settings) -> RagService:
    """Service RAG avec le registre SQL si app.db existe, sinon JSON (CLI)."""
    if settings.app_db_path.exists():
        from rag_builder.db.session import make_engine
        from rag_builder.db.sql_registry import SqlCollectionRegistry

        registry = SqlCollectionRegistry(make_engine(settings.app_db_path))
        return RagService.from_settings(settings, registry=registry)
    return RagService.from_settings(settings)


def _load_golden(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(json.loads(line))
    return items


def evaluate(golden: list[dict], svc: RagService, *, generate: bool) -> dict:
    per_item = []
    for item in golden:
        collection = item["collection"]
        question = item["question"]
        expected_doc = item.get("expected_doc")
        keywords = [k.lower() for k in item.get("expected_keywords", [])]

        result = svc.retrieve(collection, question)
        sources = [s["source_name"] for s in result.sources()][:5]

        # Rang (1-based) du premier document attendu.
        rank = next((i + 1 for i, s in enumerate(sources) if s == expected_doc), None)
        recall5 = 1.0 if rank is not None else 0.0
        mrr = 1.0 / rank if rank else 0.0

        kw_rate: float | None = None
        answer = ""
        if generate and keywords:
            try:
                answer = "".join(svc.stream_answer(collection, question, result)).lower()
                kw_rate = sum(1 for k in keywords if k in answer) / len(keywords)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ génération indisponible ({exc}) — mots-clés ignorés", file=sys.stderr)

        per_item.append(
            {
                "question": question,
                "collection": collection,
                "expected_doc": expected_doc,
                "retrieved": sources,
                "rank": rank,
                "recall@5": recall5,
                "mrr": mrr,
                "keyword_rate": kw_rate,
                "timings_ms": result.timings.as_dict(),
            }
        )

    n = len(per_item) or 1
    kw_vals = [x["keyword_rate"] for x in per_item if x["keyword_rate"] is not None]
    summary = {
        "n_questions": len(per_item),
        "recall@5": round(sum(x["recall@5"] for x in per_item) / n, 4),
        "mrr": round(sum(x["mrr"] for x in per_item) / n, 4),
        "keyword_presence": round(sum(kw_vals) / len(kw_vals), 4) if kw_vals else None,
    }
    return {"summary": summary, "items": per_item}


def _print_table(report: dict) -> None:
    print(f"\n{'#':>2}  {'R@5':>4}  {'MRR':>5}  {'kw':>5}  question")
    print("-" * 72)
    for i, it in enumerate(report["items"], 1):
        kw = "—" if it["keyword_rate"] is None else f"{it['keyword_rate']:.2f}"
        print(f"{i:>2}  {it['recall@5']:>4.0f}  {it['mrr']:>5.2f}  {kw:>5}  {it['question'][:48]}")
    s = report["summary"]
    print("-" * 72)
    kw = "—" if s["keyword_presence"] is None else f"{s['keyword_presence']:.2f}"
    print(f"    R@5={s['recall@5']:.3f}  MRR={s['mrr']:.3f}  présence mots-clés={kw}  "
          f"(n={s['n_questions']})\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Évaluation du RAG (recall@5, MRR, mots-clés)")
    parser.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    parser.add_argument("--no-generate", action="store_true", help="Retrieval uniquement")
    parser.add_argument("--output", default=str(ROOT / "eval" / "report.json"))
    args = parser.parse_args(argv)

    settings = get_settings()
    svc = _build_service(settings)
    try:
        golden = _load_golden(Path(args.golden))
        report = evaluate(golden, svc, generate=not args.no_generate)
    finally:
        svc.close()

    _print_table(report)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport JSON écrit dans {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
