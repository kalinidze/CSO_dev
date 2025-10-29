#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import math
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.request
import urllib.error


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, body
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e}")


def call_openai_embeddings(api_key: str, base_url: Optional[str], model: str, inputs: List[str]) -> Tuple[int, Optional[Dict[str, Any]]]:
    url_base = (base_url.rstrip("/") if base_url else "https://api.openai.com")
    url = url_base + "/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "input": inputs,
    }
    status, body = http_post_json(url, payload, headers)
    if status != 200:
        try:
            return status, json.loads(body)
        except Exception:
            return status, {"message": body[:500]}
    try:
        return status, json.loads(body)
    except Exception as e:
        return status, {"message": f"parse_error: {e}"}


def resource_hash(res_type: str) -> str:
    return hashlib.sha1(res_type.encode("utf-8")).hexdigest()[:16]


def build_resource_text(res_type: str, enriched: Dict[str, Any]) -> str:
    rec = enriched.get(res_type, {}) if enriched else {}
    parts: List[str] = [res_type]
    if rec:
        cat = rec.get("canonical_category") or ""
        summary = rec.get("resource_summary") or ""
        aliases = rec.get("aliases") or []
        parts.append(f"category: {cat}")
        parts.append(f"summary: {summary}")
        if isinstance(aliases, list) and aliases:
            parts.append("aliases: " + ", ".join(aliases))
    return " \n".join(parts)


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def render_progress(prefix: str, done: int, total: int, width: int = 40) -> None:
    total = max(1, total)
    frac = min(1.0, max(0.0, done / total))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix}: [{bar}] {done}/{total}", end="", file=sys.stderr)
    if done >= total:
        print("", file=sys.stderr)


def cluster_cosine_graph(embeds: Dict[str, List[float]], threshold: float, max_cluster_size: int, show_progress: bool) -> List[List[str]]:
    keys = list(embeds.keys())
    neighbors: Dict[str, set] = {k: set() for k in keys}
    total_outer = len(keys)
    for i, a in enumerate(keys):
        va = embeds[a]
        for j in range(i + 1, len(keys)):
            b = keys[j]
            vb = embeds[b]
            s = cosine(va, vb)
            if s >= threshold:
                neighbors[a].add(b)
                neighbors[b].add(a)
        if show_progress:
            render_progress("Clustering (adjacency)", i + 1, total_outer)
    visited = set()
    comps: List[List[str]] = []
    for node in keys:
        if node in visited:
            continue
        stack = [node]
        comp = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neighbors[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comp.sort()
        comps.append(comp)
    if max_cluster_size and max_cluster_size > 0:
        split: List[List[str]] = []
        for comp in comps:
            if len(comp) <= max_cluster_size:
                split.append(comp)
                continue
            remaining = set(comp)
            degree = {n: len(neighbors[n] & remaining) for n in remaining}
            while remaining:
                seed = max(remaining, key=lambda n: degree.get(n, 0))
                grp = [seed]
                remaining.remove(seed)
                added = True
                while added and len(grp) < max_cluster_size:
                    added = False
                    cand_list = list(remaining)
                    best = None
                    best_deg = -1
                    for c in cand_list:
                        if all(c in neighbors[m] for m in grp):
                            d = len(neighbors[c] & remaining)
                            if d > best_deg:
                                best_deg = d
                                best = c
                    if best is not None:
                        grp.append(best)
                        remaining.remove(best)
                        added = True
                if len(grp) > 1:
                    split.append(sorted(grp))
                degree = {n: len(neighbors[n] & remaining) for n in remaining}
        comps = split
    comps = [c for c in comps if len(c) > 1]
    return comps


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute OpenAI embeddings for resources and cluster them.",
    )
    parser.add_argument(
        "--providers-dir",
        default=os.path.join(os.getcwd(), "providers"),
        help="Directory containing enriched_descriptions.json and outputs (default: ./providers)",
    )
    parser.add_argument(
        "--enriched",
        default=None,
        help="Path to enriched_descriptions.json (default: <providers-dir>/enriched_descriptions.json)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (env OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Custom base URL (optional)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        help="OpenAI embedding model (default: text-embedding-3-large)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent embedding batches (default: 4)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between batches per worker (default: 0.1)",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.55,
        help="Cosine similarity threshold for edges (default: 0.55)",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=6,
        help="Maximum cluster size after splitting (default: 6)",
    )
    parser.add_argument(
        "--limit-resources",
        type=int,
        default=None,
        help="Limit number of resources (optional)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for embeddings (default: <providers-dir>/enrichment_cache/embeddings)",
    )
    parser.add_argument(
        "--embeddings-out",
        default=None,
        help="Output JSONL for resource embeddings (default: <providers-dir>/resource_embeddings.jsonl)",
    )
    parser.add_argument(
        "--top-similar-out",
        default=None,
        help="Output JSON for embedding-based top similar (default: <providers-dir>/embedding_top_similar.json)",
    )
    parser.add_argument(
        "--clusters-out",
        default=None,
        help="Output JSON for embedding-based clusters (default: <providers-dir>/embedding_clusters.json)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    enriched_path = args.enriched or os.path.join(args.providers_dir, "enriched_descriptions.json")
    cache_dir = args.cache_dir or os.path.join(args.providers_dir, "enrichment_cache", "embeddings")
    embeds_out = args.embeddings_out or os.path.join(args.providers_dir, "resource_embeddings.jsonl")
    top_out = args.top_similar_out or os.path.join(args.providers_dir, "embedding_top_similar.json")
    clusters_out = args.clusters_out or os.path.join(args.providers_dir, "embedding_clusters.json")

    if not args.api_key:
        print("Missing OpenAI API key.", file=sys.stderr)
        return 2

    enriched = read_json(enriched_path) or {}
    resources = list(enriched.keys())
    if args.limit_resources is not None:
        resources = resources[: max(0, args.limit_resources)]

    ensure_dir(cache_dir)

    embeddings: Dict[str, List[float]] = {}
    errors: Dict[str, Any] = {}

    def embed_batch(batch: List[str]) -> List[Tuple[str, Any]]:
        texts = [build_resource_text(r, enriched) for r in batch]
        status, resp = call_openai_embeddings(args.api_key or "", args.base_url, args.embedding_model, texts)
        results: List[Tuple[str, Any]] = []
        if status != 200 or not isinstance(resp, dict) or "data" not in resp:
            for r in batch:
                results.append((r, {"_error": resp}))
            return results
        datas = resp.get("data", [])
        if len(datas) != len(batch):
            for r in batch:
                results.append((r, {"_error": "mismatched_batch"}))
            return results
        for r, row in zip(batch, datas):
            vec = row.get("embedding")
            results.append((r, vec))
        return results

    to_embed: List[str] = []
    for r in resources:
        cache_file = os.path.join(cache_dir, f"{resource_hash(r)}.json")
        cached = read_json(cache_file)
        if isinstance(cached, list):
            embeddings[r] = cached
        else:
            to_embed.append(r)

    if to_embed:
        batch_size = max(1, args.batch_size)
        batches = [to_embed[i:i + batch_size] for i in range(0, len(to_embed), batch_size)]
        total_batches = len(batches)
        done_batches = 0
        if not args.no_progress:
            render_progress("Embeddings", 0, total_batches)
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [executor.submit(embed_batch, b) for b in batches]
            for fut in as_completed(futures):
                res = fut.result()
                for r, vec_or_err in res:
                    cache_file = os.path.join(cache_dir, f"{resource_hash(r)}.json")
                    if isinstance(vec_or_err, list):
                        embeddings[r] = vec_or_err
                        write_json(cache_file, vec_or_err)
                    else:
                        errors[r] = vec_or_err
                done_batches += 1
                if not args.no_progress:
                    render_progress("Embeddings", done_batches, total_batches)
                time.sleep(max(0.0, args.sleep))
        if not args.no_progress and done_batches < total_batches:
            render_progress("Embeddings", total_batches, total_batches)

    rows = []
    for r, vec in embeddings.items():
        rows.append({"resource": r, "embedding": vec})
    write_jsonl(embeds_out, rows)

    keys = list(embeddings.keys())
    top_sim: Dict[str, List[Tuple[str, float]]] = {k: [] for k in keys}
    total_outer = len(keys)
    if not args.no_progress:
        render_progress("Top-sim", 0, total_outer)
    for i, a in enumerate(keys):
        va = embeddings[a]
        scored: List[Tuple[str, float]] = []
        for j, b in enumerate(keys):
            if i == j:
                continue
            vb = embeddings[b]
            s = cosine(va, vb)
            if s <= 0:
                continue
            scored.append((b, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sim[a] = scored[:10]
        if not args.no_progress:
            render_progress("Top-sim", i + 1, total_outer)
    write_json(top_out, top_sim)

    clusters = cluster_cosine_graph(embeddings, threshold=args.sim_threshold, max_cluster_size=args.max_cluster_size, show_progress=not args.no_progress)
    write_json(clusters_out, clusters)

    print(f"Wrote {embeds_out}")
    print(f"Wrote {top_out}")
    print(f"Wrote {clusters_out}")
    if errors:
        print(f"Embedding errors for {len(errors)} resources (see cache)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
