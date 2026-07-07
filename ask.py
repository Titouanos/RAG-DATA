#!/usr/bin/env python3
"""CLI pour poser des questions au RAG.

Usage:
    python ask.py                                      # mode interactif
    python ask.py "ma question"                        # one-shot
    python ask.py -v "ma question"                     # avec debug retrieval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag.config import load_config
from rag.pipeline import build_components, query


def render_answer(result, show_chunks: bool = False):
    """Formatage de la réponse pour le terminal."""
    print()
    print("=" * 70)
    print(result.answer)
    print("=" * 70)

    if result.sources:
        print()
        print("Sources :")
        for s in result.sources:
            source_str = s["source"] or ""
            # Raccourcir les chemins longs
            if len(source_str) > 70:
                source_str = "..." + source_str[-67:]
            print(f"  [{s['num']}] ({s['type']}) {s['title']}")
            if source_str:
                print(f"       {source_str}")

    if show_chunks and result.used_chunks:
        print()
        print("--- Chunks utilisés ---")
        for i, c in enumerate(result.used_chunks, 1):
            preview = c.text[:200].replace("\n", " ")
            print(f"[{i}] {preview}...")
    print()


def interactive(components):
    """Mode interactif : pose des questions en boucle."""
    print()
    print("Mode interactif — tape ta question (ou 'quit' pour sortir)")
    print()
    while True:
        try:
            question = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q", ":q"}:
            break
        try:
            result = query(components, question)
            render_answer(result)
        except Exception as exc:
            print(f"Erreur : {exc}")


def main():
    parser = argparse.ArgumentParser(description="Poser une question au RAG")
    parser.add_argument("question", nargs="?", help="Question (vide = interactif)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Affiche les scores et chunks récupérés")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Affiche les chunks utilisés pour la réponse")
    parser.add_argument("--config", default="config.yaml",
                        help="Chemin du fichier de config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ["httpx", "chromadb", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        config = load_config(args.config)
    except RuntimeError as exc:
        print(f"ERREUR CONFIG : {exc}", file=sys.stderr)
        sys.exit(1)

    components = build_components(config)

    # Vérifier qu'il y a au moins 1 chunk
    if components.vector_store.count() == 0:
        print("⚠️  La base est vide. Lance d'abord : python ingest.py", file=sys.stderr)
        sys.exit(2)

    if args.question:
        result = query(components, args.question, verbose=args.verbose)
        render_answer(result, show_chunks=args.show_chunks or args.verbose)
    else:
        interactive(components)


if __name__ == "__main__":
    main()
