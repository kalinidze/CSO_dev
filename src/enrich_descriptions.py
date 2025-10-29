#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import math
import hashlib
import re
import random
from typing import Dict, Any, Optional, List, Tuple

import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_property_idf(properties_index: Dict[str, List[Dict[str, Any]]], total_resources: int) -> Dict[str, float]:
    idf: Dict[str, float] = {}
    for fid, occ in properties_index.items():
        df = len(occ) if isinstance(occ, list) else 1
        idf[fid] = math.log((1 + total_resources) / (1 + df)) + 1.0
    return idf


def select_properties_for_resource(resource_props: Dict[str, Dict[str, Any]], idf: Dict[str, float], max_props: int) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any], float, int]] = []
    for path, meta in resource_props.items():
        fid = meta.get("id")
        weight = idf.get(fid, 1.0)
        depth = path.count(".")
        score = weight - 0.05 * depth
        items.append((path, meta, score, depth))
    items.sort(key=lambda x: (x[2], -x[3]), reverse=True)
    selected = [(p, m) for p, m, _s, _d in items[:max_props]]
    return selected


def provider_from_resource(res_type: str) -> str:
    idx = res_type.find("_")
    return res_type[:idx] if idx > 0 else ""


def build_prompt(res_type: str, res_desc: str, props: List[Tuple[str, Dict[str, Any]]]) -> str:
    provider = provider_from_resource(res_type)

    lines: List[str] = []
    lines.append("You are assisting in mapping semantically similar cloud resources across Terraform providers (e.g., Yandex Lockbox ↔ AWS Secrets Manager).")
    lines.append("Goal: produce enriched, provider-agnostic descriptions to help identify equivalent resources in other providers.")
    lines.append("Output STRICT JSON only matching the schema below. Do not include any commentary.")
    lines.append("")
    lines.append("Input:")
    lines.append(f"resource_type: {res_type}")
    lines.append(f"provider: {provider}")
    lines.append("resource_description: |-")
    if res_desc:
        for ln in res_desc.splitlines():
            lines.append(f"  {ln}")
    else:
        lines.append("  ")
    lines.append("properties:")
    for path, meta in props:
        desc = meta.get("description") or ""
        lines.append(f"- path: {path}")
        lines.append(f"  name: {path.split('.')[-1]}")
        lines.append("  description: |-")
        if desc:
            for ln in desc.splitlines():
                lines.append(f"    {ln}")
        else:
            lines.append("    ")
    lines.append("")
    lines.append("Return JSON with this schema:")
    lines.append("{")
    lines.append("  \"resource_summary\": string,  // concise, provider-agnostic purpose")
    lines.append("  \"canonical_category\": string, // e.g., 'Secrets Management', 'Object Storage'")
    lines.append("  \"aliases\": [string],       // common product names across providers")
    lines.append("  \"properties\": {")
    lines.append("    \"<path>\": { \"canonical_name\": string, \"purpose\": string, \"synonyms\": [string], \"enriched_description\": string }")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def call_openai(api_key: str, model: str, base_url: Optional[str], prompt: str, verbose: bool, log_file: Optional[str], max_retries: int = 3) -> Optional[Dict[str, Any]]:
    url = (base_url.rstrip("/") if base_url else "https://api.openai.com") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise AI that outputs strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, max_retries + 1):
        status, body = http_post_json(url, payload, headers)
        if verbose:
            msg = f"OpenAI attempt {attempt}: HTTP {status}"
            print(msg, file=sys.stderr)
            if log_file:
                ensure_dir(os.path.dirname(log_file))
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(msg + "\n")
        if status == 200:
            try:
                resp = json.loads(body)
                content = resp.get("choices", [{}])[0].get("message", {}).get("content")
                if not content:
                    err = {"_error": "empty_content", "_http_status": status}
                    if verbose:
                        print("OpenAI: empty content", file=sys.stderr)
                    if log_file:
                        with open(log_file, "a", encoding="utf-8") as lf:
                            lf.write("empty content\n")
                    return err
                return json.loads(content)
            except Exception as e:
                err = {"_error": "parse_error", "_http_status": status, "_detail": str(e)}
                if verbose:
                    print(f"OpenAI: parse error: {e}", file=sys.stderr)
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as lf:
                        lf.write(f"parse error: {e}\n")
                return err
        else:
            # Backoff for 429/5xx
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(0.5 * attempt)
                continue
            try:
                err_json = json.loads(body)
            except Exception:
                err_json = {"message": body[:500]}
            err = {"_error": "http_error", "_http_status": status, "_body": err_json}
            if verbose:
                print(f"OpenAI HTTP {status}: {str(err_json)[:500]}", file=sys.stderr)
            if log_file:
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"HTTP {status}: {str(err_json)[:500]}\n")
            return err
    return {"_error": "exhausted_retries"}


def hash_resource(res_type: str) -> str:
    return hashlib.sha1(res_type.encode("utf-8")).hexdigest()[:16]


def log_msg(verbose: bool, log_file: Optional[str], message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(message + "\n")


def enrich_resources(
    annotated: Dict[str, Any],
    properties_index: Dict[str, List[Dict[str, Any]]],
    api_key: str,
    model: str,
    base_url: Optional[str],
    out_cache_dir: str,
    limit_resources: Optional[int],
    max_props_per_resource: int,
    sleep_seconds: float,
    dry_run: bool,
    force: bool,
    resource_prefixes: Optional[List[str]],
    sample_random: bool,
    random_seed: Optional[int],
    verbose: bool,
    log_file: Optional[str],
    concurrency: int,
) -> Dict[str, Any]:
    total_resources = len(annotated)
    idf = compute_property_idf(properties_index, total_resources)

    enriched: Dict[str, Any] = {}

    selected_resources: List[str] = list(annotated.keys())
    if resource_prefixes:
        prfx = tuple(resource_prefixes)
        selected_resources = [r for r in selected_resources if r.startswith(prfx)]

    if sample_random:
        rng = random.Random(random_seed)
        rng.shuffle(selected_resources)

    pre_limit_count = len(selected_resources)
    if limit_resources is not None:
        selected_resources = selected_resources[: max(0, limit_resources)]

    log_msg(verbose, log_file, f"Selected resources: {len(selected_resources)} (from {pre_limit_count}); random={sample_random}, limit={limit_resources}, prefixes={'yes' if resource_prefixes else 'no'}, concurrency={concurrency}")

    ensure_dir(out_cache_dir)

    def process_one(res_type: str) -> Tuple[str, Dict[str, Any]]:
        res = annotated[res_type]
        cache_path = os.path.join(out_cache_dir, f"{hash_resource(res_type)}.json")
        if not force and os.path.exists(cache_path):
            cached = read_json(cache_path)
            if cached is not None:
                return res_type, cached
        props_dict: Dict[str, Dict[str, Any]] = res.get("properties") or {}
        if not props_dict:
            return res_type, {"_error": "no_properties"}
        selected_props = select_properties_for_resource(props_dict, idf, max_props_per_resource)
        prompt = build_prompt(res_type, res.get("description") or "", selected_props)
        if dry_run:
            enriched_obj = {
                "resource_summary": "",
                "canonical_category": "",
                "aliases": [],
                "properties": {p: {"canonical_name": "", "purpose": "", "synonyms": [], "enriched_description": ""} for p, _ in selected_props},
                "_note": "dry_run",
            }
        else:
            resp = call_openai(api_key, model, base_url, prompt, verbose=verbose, log_file=log_file)
            if not isinstance(resp, dict) or resp.get("_error"):
                enriched_obj = resp if isinstance(resp, dict) else {"_error": "api_call_failed"}
            else:
                enriched_obj = resp
        write_json(cache_path, enriched_obj)
        if not dry_run and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return res_type, enriched_obj

    if concurrency <= 1:
        for idx, res_type in enumerate(selected_resources, 1):
            log_msg(verbose, log_file, f"[{idx}/{len(selected_resources)}] Processing {res_type}")
            r, obj = process_one(res_type)
            enriched[r] = obj
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_res = {}
            for res_type in selected_resources:
                future = executor.submit(process_one, res_type)
                future_to_res[future] = res_type
            completed = 0
            for future in as_completed(future_to_res):
                res_type = future_to_res[future]
                try:
                    r, obj = future.result()
                    enriched[r] = obj
                    completed += 1
                    if completed % max(1, len(selected_resources)//10) == 0 or verbose:
                        log_msg(verbose, log_file, f"Completed {completed}/{len(selected_resources)}: {res_type}")
                except Exception as e:
                    log_msg(True, log_file, f"Error processing {res_type}: {e}")

    log_msg(verbose, log_file, f"Completed enrichment for {len(enriched)} resources")
    return enriched


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich resource and property descriptions via OpenAI to aid cross-provider mapping.",
    )
    parser.add_argument(
        "--providers-dir",
        default=os.path.join(os.getcwd(), "providers"),
        help="Directory containing combined_resources_annotated.json and properties.json (default: ./providers)",
    )
    parser.add_argument(
        "--annotated",
        default=None,
        help="Path to combined_resources_annotated.json (default: <providers-dir>/combined_resources_annotated.json)",
    )
    parser.add_argument(
        "--properties-index",
        default=None,
        help="Path to properties.json (default: <providers-dir>/properties.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON for enriched descriptions (default: <providers-dir>/enriched_descriptions.json)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache per-resource enrichment (default: <providers-dir>/enrichment_cache/resources)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (default from OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Custom OpenAI base URL (optional)",
    )
    parser.add_argument(
        "--max-props-per-resource",
        type=int,
        default=20,
        help="Max properties to include per resource prompt (default: 20)",
    )
    parser.add_argument(
        "--limit-resources",
        type=int,
        default=None,
        help="Limit number of resources to process (optional)",
    )
    parser.add_argument(
        "--resource-prefixes",
        nargs="+",
        default=None,
        help="Only process resources starting with any of these prefixes (e.g., aws_ google_)",
    )
    parser.add_argument(
        "--sample-random",
        action="store_true",
        help="Randomly sample resources before applying --limit-resources",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for --sample-random (optional)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between API calls (default: 0.2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call API; produce empty-shaped outputs for inspection",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-enrich",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose diagnostics and API errors to stderr",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path to write verbose diagnostics (default: <providers-dir>/enrichment_cache/enrich.log)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    annotated_path = args.annotated or os.path.join(args.providers_dir, "combined_resources_annotated.json")
    props_index_path = args.properties_index or os.path.join(args.providers_dir, "properties.json")
    out_path = args.output or os.path.join(args.providers_dir, "enriched_descriptions.json")
    cache_dir = args.cache_dir or os.path.join(args.providers_dir, "enrichment_cache", "resources")
    log_file = args.log_file or os.path.join(args.providers_dir, "enrichment_cache", "enrich.log")

    if not args.api_key and not args.dry_run:
        print("Missing OpenAI API key. Set --api-key or OPENAI_API_KEY.", file=sys.stderr)
        return 2

    annotated = read_json(annotated_path) or {}
    if not annotated:
        print(f"Annotated resources not found or empty: {annotated_path}", file=sys.stderr)
        return 2
    props_index = read_json(props_index_path) or {}

    enriched = enrich_resources(
        annotated=annotated,
        properties_index=props_index,
        api_key=args.api_key or "",
        model=args.model,
        base_url=args.base_url,
        out_cache_dir=cache_dir,
        limit_resources=args.limit_resources,
        max_props_per_resource=args.max_props_per_resource,
        sleep_seconds=args.sleep,
        dry_run=args.dry_run,
        force=args.force,
        resource_prefixes=args.resource_prefixes,
        sample_random=args.sample_random,
        random_seed=args.random_seed,
        verbose=args.verbose,
        log_file=log_file,
        concurrency=max(1, args.concurrency),
    )

    write_json(out_path, enriched)
    print(f"Wrote {out_path}")
    print(f"Logs: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
