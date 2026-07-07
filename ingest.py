#!/usr/bin/env python3
"""CLI d'ingestion.

Usage:
    python ingest.py                    # ingestion incrémentale
    python ingest.py --reset            # vide et ré-ingère tout
    python ingest.py --stats            # affiche l'état de la base
    python ingest.py --verbose          # logs + preview des chunks
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.config import load_config
from rag.pipeline import build_components, ingest, print_stats


def main():
    parser = argparse.ArgumentParser(description="Ingestion du RAG")
    parser.add_argument("--reset", action="store_true",
                        help="Vide la base avant ingestion")
    parser.add_argument("--stats", action="store_true",
                        help="Affiche l'état de la base et quitte")
    parser.add_argument("--quota", action="store_true",
                        help="Affiche les quotas Gemini utilisés aujourd'hui et quitte")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Logs détaillés")
    parser.add_argument("--config", default="config.yaml",
                        help="Chemin du fichier de config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Pour éviter le bruit de chromadb/httpx
    for noisy in ["httpx", "chromadb", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        config = load_config(args.config)
    except RuntimeError as exc:
        print(f"ERREUR CONFIG : {exc}", file=sys.stderr)
        sys.exit(1)

    components = build_components(config)

    if args.quota:
        print("=== Quotas Gemini utilisés aujourd'hui (cumul depuis reset) ===")
        stats = components.rate_limiter.stats()
        if not stats:
            print("(aucun appel enregistré sur cette journée — les compteurs "
                  "démarrent dès le premier appel)")
        else:
            print(f"{'Modèle':<30} {'RPM':>10}  {'RPD':>15}  {'Reset dans':>12}")
            print("-" * 80)
            for model, s in stats.items():
                h = s["rpd_reset_in_seconds"] // 3600
                m = (s["rpd_reset_in_seconds"] % 3600) // 60
                print(f"{model:<30} "
                      f"{s['rpm_used']:>4}/{s['rpm_limit']:<4}  "
                      f"{s['rpd_used']:>6}/{s['rpd_limit']:<6}  "
                      f"{h:>2}h {m:>2}min")
        return

    if args.stats:
        print_stats(components)
        return

    print(f"Data dir    : {config.data_dir}")
    print(f"Storage dir : {config.storage_dir}")
    print(f"Model emb   : {config.embedding_model} ({config.embedding_dim} dims)")
    print(f"Model gen   : {config.generation_model}")
    print(f"Reranker    : {'activé' if config.use_reranker else 'désactivé'}")
    print()

    report = ingest(components, reset=args.reset, verbose=args.verbose)

    print()
    print("=== Résumé ingestion ===")
    print(f"Sources scannées  : {report.total_sources}")
    print(f"Nouveaux docs     : {report.new_docs}")
    print(f"Docs mis à jour   : {report.updated_docs}")
    print(f"Docs inchangés    : {report.unchanged_docs}")
    print(f"Docs supprimés    : {report.deleted_docs}")
    print(f"Échecs            : {report.failed_sources}")
    print(f"Chunks indexés    : {report.total_chunks}")
    if report.errors:
        print()
        print("Erreurs :")
        for err in report.errors[:10]:
            print(f"  - {err}")
        if len(report.errors) > 10:
            print(f"  ... et {len(report.errors) - 10} de plus")


if __name__ == "__main__":
    main()
