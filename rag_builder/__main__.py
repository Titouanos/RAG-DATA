"""CLI de validation du cœur RAG Builder (Phase 1).

    python -m rag_builder create-collection NAME [--description ...] [--no-rerank]
    python -m rag_builder list-collections
    python -m rag_builder ingest --collection NAME [PATH ...]      # défaut : data_dir
    python -m rag_builder ask --collection NAME "question" [-v] [--no-generate]
    python -m rag_builder delete --collection NAME (--doc-id ID | --source NOM)
    python -m rag_builder stats --collection NAME
    python -m rag_builder delete-collection NAME [--yes]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_builder.config import get_settings
from rag_builder.core.converters.base import iter_sources, make_doc_id
from rag_builder.core.rag_service import RagService
from rag_builder.core.registry import CollectionError
from rag_builder.logging_conf import setup_logging


def _svc() -> RagService:
    return RagService.from_settings(get_settings())


def cmd_create_collection(args) -> int:
    svc = _svc()
    try:
        overrides = {}
        if args.embedder:
            overrides["embedder"] = args.embedder
        if args.no_rerank:
            overrides["rerank_enabled"] = False
        if args.top_k is not None:
            overrides["top_k"] = args.top_k
        if args.llm_provider:
            overrides["llm_provider"] = args.llm_provider
        if args.llm_model:
            overrides["llm_model"] = args.llm_model
        meta = svc.create_collection(args.name, description=args.description or "", **overrides)
        print(f"Collection '{meta.name}' créée (embedding={meta.embedding_model}, "
              f"sparse={meta.supports_sparse}, rerank={meta.rerank_enabled}, top_k={meta.top_k}).")
        return 0
    except CollectionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        svc.close()


def cmd_list_collections(args) -> int:
    svc = _svc()
    try:
        cols = svc.registry.list()
        if not cols:
            print("(aucune collection)")
            return 0
        print(f"{'Collection':<24} {'Embedding':<16} {'Chunks':>7} {'Rerank':>7}  Description")
        print("-" * 90)
        for m in cols:
            n = svc.store.count(m.name)
            print(f"{m.name:<24} {m.embedding_model.split('/')[-1]:<16} {n:>7} "
                  f"{'oui' if m.rerank_enabled else 'non':>7}  {m.description[:40]}")
        return 0
    finally:
        svc.close()


def cmd_ingest(args) -> int:
    svc = _svc()
    try:
        svc.registry.require(args.collection)
        paths: list[Path] = []
        if args.paths:
            for p in args.paths:
                paths.append(Path(p))
        else:
            paths = list(iter_sources(svc.settings.data_dir))
        if not paths:
            print("Aucun fichier à ingérer (précise des chemins ou remplis data/).",
                  file=sys.stderr)
            return 2
        counts = {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
        total_chunks = 0
        for p in paths:
            res = svc.ingest_document(args.collection, p)
            counts[res.status] = counts.get(res.status, 0) + 1
            total_chunks += res.n_chunks
            tag = res.status.upper()
            extra = f" ({res.n_chunks} chunks)" if res.n_chunks else ""
            msg = f" — {res.message}" if res.message else ""
            print(f"[{tag:<7}] {res.source_name}{extra}{msg}")
        print(f"\nRésumé : {counts['new']} nouveaux, {counts['updated']} mis à jour, "
              f"{counts['skipped']} inchangés, {counts['failed']} échecs, "
              f"{total_chunks} chunks indexés.")
        return 0
    except CollectionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        svc.close()


def cmd_ask(args) -> int:
    svc = _svc()
    try:
        result = svc.retrieve(args.collection, args.question)
        if not result.chunks:
            print("Aucun extrait pertinent trouvé.", file=sys.stderr)
            return 0

        if args.verbose:
            print("\n--- Chunks retrouvés ---")
            for s in result.sources():
                print(f"[{s['n']}] score={s['score']} — {s['source_name']} "
                      f"{s['page_or_section']}")
                print(f"     {s['excerpt'][:160].strip()}…")

        if not args.no_generate:
            print("\n--- Réponse ---")
            for token in svc.stream_answer(args.collection, args.question, result):
                sys.stdout.write(token)
                sys.stdout.flush()
            print()

        print("\nSources :")
        for s in result.sources():
            print(f"  [{s['n']}] {s['source_name']} {s['page_or_section']} (score {s['score']})")
        print(f"\nLatence : {result.timings.as_dict()}")
        return 0
    except CollectionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Erreur à la génération : {exc}", file=sys.stderr)
        return 1
    finally:
        svc.close()


def cmd_delete(args) -> int:
    svc = _svc()
    try:
        doc_id = args.doc_id or make_doc_id(args.source)
        n = svc.delete_document(args.collection, doc_id)
        print(f"Document {doc_id} supprimé ({n} chunks).")
        return 0
    except CollectionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        svc.close()


def cmd_stats(args) -> int:
    svc = _svc()
    try:
        meta = svc.registry.require(args.collection)
        doc_ids = svc.store.list_doc_ids(args.collection)
        print(f"Collection    : {meta.name}")
        print(f"Embedding     : {meta.embedding_model} ({meta.dense_dim}d, "
              f"sparse={meta.supports_sparse})")
        print(f"Rerank        : {'activé' if meta.rerank_enabled else 'désactivé'} "
              f"({meta.rerank_model})")
        print(f"Documents     : {len(doc_ids)}")
        print(f"Chunks        : {svc.store.count(args.collection)}")
        return 0
    except CollectionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        svc.close()


def cmd_create_user(args) -> int:
    import getpass

    from rag_builder.api.auth import create_user
    from rag_builder.db.models import Role
    from rag_builder.db.session import init_db, make_engine, session_scope

    settings = get_settings()
    settings.ensure_dirs()
    engine = make_engine(settings.app_db_path)
    init_db(engine)
    password = args.password or getpass.getpass("Mot de passe : ")
    if not password:
        print("Mot de passe requis.", file=sys.stderr)
        return 1
    role = Role.ADMIN if args.admin else (args.role or Role.USER)
    try:
        with session_scope(engine) as db:
            user = create_user(db, args.username, password, role=role)
            username, user_role = user.username, user.role  # avant fermeture de session
        print(f"Utilisateur '{username}' créé (rôle {user_role}).")
        return 0
    except ValueError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "rag_builder.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_delete_collection(args) -> int:
    svc = _svc()
    try:
        if not svc.registry.exists(args.name):
            print(f"Collection introuvable : {args.name}", file=sys.stderr)
            return 1
        if not args.yes:
            print(f"Confirme la suppression de '{args.name}' avec --yes.", file=sys.stderr)
            return 1
        svc.delete_collection(args.name)
        print(f"Collection '{args.name}' supprimée.")
        return 0
    finally:
        svc.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_builder", description="Cœur RAG Builder — CLI")
    # Parent partagé : -v/--verbose disponible avant OU après la sous-commande.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="Logs détaillés")
    p.add_argument("-v", "--verbose", action="store_true", help="Logs détaillés")
    sub = p.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)

    def add(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    c = add("create-collection", help="Crée une collection")
    c.add_argument("name")
    c.add_argument("--description", default="")
    c.add_argument("--embedder", choices=["local_bge_m3", "gemini"], default=None)
    c.add_argument("--no-rerank", action="store_true")
    c.add_argument("--top-k", type=int, default=None)
    c.add_argument("--llm-provider", default=None)
    c.add_argument("--llm-model", default=None)
    c.set_defaults(func=cmd_create_collection)

    lc = add("list-collections", help="Liste les collections")
    lc.set_defaults(func=cmd_list_collections)

    ing = add("ingest", help="Ingère des documents")
    ing.add_argument("--collection", required=True)
    ing.add_argument("paths", nargs="*", help="Fichiers (défaut : contenu de data/)")
    ing.set_defaults(func=cmd_ingest)

    a = add("ask", help="Interroge une collection")
    a.add_argument("--collection", required=True)
    a.add_argument("question")
    a.add_argument("--no-generate", action="store_true", help="Retrieval seul, pas de LLM")
    a.set_defaults(func=cmd_ask)

    d = add("delete", help="Supprime un document")
    d.add_argument("--collection", required=True)
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc-id")
    g.add_argument("--source", help="Nom de fichier source (doc_id dérivé)")
    d.set_defaults(func=cmd_delete)

    s = add("stats", help="État d'une collection")
    s.add_argument("--collection", required=True)
    s.set_defaults(func=cmd_stats)

    dc = add("delete-collection", help="Supprime une collection entière")
    dc.add_argument("name")
    dc.add_argument("--yes", action="store_true")
    dc.set_defaults(func=cmd_delete_collection)

    cu = add("create-user", help="Crée un compte (API)")
    cu.add_argument("username")
    cu.add_argument("--password", default=None, help="Sinon demandé interactivement")
    cu.add_argument("--admin", action="store_true")
    cu.add_argument("--role", default=None)
    cu.set_defaults(func=cmd_create_user)

    sv = add("serve", help="Lance l'API HTTP (uvicorn)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
