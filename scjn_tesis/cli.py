"""CLI: python -m scjn_tesis --help"""

from __future__ import annotations

import argparse
import sys

from scjn_tesis.buscador_juridico import scrape_buscador_juridico
from scjn_tesis.models import SearchParams, save_json
from scjn_tesis.semanario import scrape_semanario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Scraper SCJN: Buscador Jurídico (bj) o Semanario Judicial (sjf).",
    )
    p.add_argument(
        "--source",
        choices=("bj", "sjf"),
        required=True,
        help="bj = bj.scjn.gob.mx | sjf = sjf2.scjn.gob.mx",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-q",
        "--query",
        help='Término general (palabra clave, registro o rubro).',
    )
    g.add_argument("--registro", help="Priorizar búsqueda por número de registro digital.")
    g.add_argument("--rubro", help="Priorizar búsqueda por texto de rubro.")
    p.add_argument(
        "--out",
        "-o",
        default="tesis_scjn.json",
        help="Archivo JSON de salida.",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Máximo de páginas de resultados a recorrer (por defecto 3).",
    )
    p.add_argument(
        "--no-detail",
        action="store_true",
        help="No abrir cada detalle (solo datos del listado; más rápido).",
    )
    p.add_argument(
        "--headed",
        action="store_true",
        help="Mostrar navegador (útil si el sitio falla en headless).",
    )
    args = p.parse_args(argv)

    if args.registro:
        params = SearchParams(numero_registro=args.registro)
        q_display = args.registro
    elif args.rubro:
        params = SearchParams(rubro=args.rubro)
        q_display = args.rubro
    elif args.query:
        params = SearchParams(texto=args.query)
        q_display = args.query
    else:
        p.error("Indica -q/--query, --registro o --rubro.")
    fetch_detail = not args.no_detail
    headless = not args.headed

    meta = {
        "source": args.source,
        "query": q_display,
        "max_pages": args.max_pages,
        "fetch_detail": fetch_detail,
        "headless": headless,
    }

    if args.source == "bj":
        items, fin = scrape_buscador_juridico(
            params,
            max_pages=args.max_pages,
            fetch_detail=fetch_detail,
            headless=headless,
        )
        if fin:
            meta["fin_paginacion"] = fin
    else:
        items = scrape_semanario(
            params,
            max_pages=args.max_pages,
            fetch_detail=fetch_detail,
            headless=headless,
        )

    save_json(args.out, items, meta=meta)
    print(f"Guardado: {args.out} ({len(items)} registros)", file=sys.stderr)
    if args.source == "bj" and meta.get("fin_paginacion"):
        print(meta["fin_paginacion"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
