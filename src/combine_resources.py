#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional, List, Tuple


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


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def summarize_markdown_intro(md: str) -> Optional[str]:
    lines = md.splitlines()
    # Find first H1 and take the paragraph after it
    i = 0
    while i < len(lines):
        if lines[i].startswith("# ") or lines[i].startswith("#\t"):
            i += 1
            break
        i += 1
    # Skip blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1
    # Collect until blank line or next section
    buf: List[str] = []
    while i < len(lines):
        if not lines[i].strip():
            break
        if lines[i].startswith("## "):
            break
        buf.append(lines[i])
        i += 1
    text = "\n".join(buf).strip()
    return text or None


def build_docs_summary_map(provider_dir: str) -> Dict[str, str]:
    docs_path = os.path.join(provider_dir, "docs_pages.jsonl")
    summaries: Dict[str, str] = {}
    for rec in read_jsonl(docs_path):
        kind = rec.get("kind")
        if kind != "resource":
            continue
        slug = rec.get("slug")
        md = rec.get("raw_markdown") or ""
        summary = summarize_markdown_intro(md)
        if slug and summary:
            summaries[slug] = summary
    return summaries


def flatten_block(block: Dict[str, Any], prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """Return mapping of dotpath -> { description } from a Terraform schema block.
    Includes attributes and nested block_types recursively.
    """
    result: Dict[str, Dict[str, Any]] = {}

    attributes = block.get("attributes", {}) or {}
    for name, attr in attributes.items():
        key = f"{prefix}.{name}" if prefix else name
        desc = attr.get("description") or ""
        result[key] = {"description": desc}

    block_types = block.get("block_types", {}) or {}
    for name, bt in block_types.items():
        nested_block = bt.get("block", {}) or {}
        nested_prefix = f"{prefix}.{name}" if prefix else name
        # Recurse
        nested_map = flatten_block(nested_block, nested_prefix)
        result.update(nested_map)

    return result


def extract_resources_from_schema(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    resources: Dict[str, Dict[str, Any]] = {}
    if not schema:
        return resources
    provider_schemas = schema.get("provider_schemas") or {}
    for _prov_addr, prov in provider_schemas.items():
        resource_schemas = prov.get("resource_schemas") or {}
        for res_type, res_schema in resource_schemas.items():
            block = res_schema.get("block", {}) or {}
            res_desc = block.get("description") or ""
            properties = flatten_block(block)
            resources[res_type] = {
                "description": res_desc,
                "properties": properties,
            }
    return resources


def slug_for_resource(provider: str, resource_type: str) -> str:
    # Remove provider prefix like aws_, google_, azurerm_, yandex_
    prefix = provider + "_"
    if resource_type.startswith(prefix):
        return resource_type[len(prefix):]
    # As a fallback, try to strip common namespaces like azurerm_
    if provider == "azurerm" and resource_type.startswith("azurerm_"):
        return resource_type[len("azurerm_"):]
    return resource_type


def combine_resources(providers_dir: str, providers: List[str]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {}

    for provider in providers:
        provider_dir = os.path.join(providers_dir, provider)
        schema_path = os.path.join(provider_dir, "schema.json")
        schema = read_json(schema_path)
        if not schema:
            continue
        res_map = extract_resources_from_schema(schema)

        # Build docs summaries for resource-level description fallback
        docs_summary = build_docs_summary_map(provider_dir)

        for res_type, res_data in res_map.items():
            # If resource description empty, try docs summary
            if not res_data.get("description"):
                slug = slug_for_resource(provider, res_type)
                if slug in docs_summary:
                    res_data["description"] = docs_summary[slug]

            # Ensure properties have description fields
            props: Dict[str, Dict[str, Any]] = res_data.get("properties") or {}
            for key, meta in props.items():
                if "description" not in meta or meta["description"] is None:
                    meta["description"] = ""

            combined[res_type] = res_data

    return combined


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-provider schemas and docs into a single resources JSON.",
    )
    parser.add_argument(
        "--providers-dir",
        default=os.path.join(os.getcwd(), "providers"),
        help="Directory containing per-provider data (default: ./providers)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <providers-dir>/combined_resources.json)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["aws", "google", "azurerm", "yandex"],
        help="Providers to include (default: aws google azurerm yandex)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    out_path = args.output or os.path.join(args.providers_dir, "combined_resources.json")
    ensure_dir(os.path.dirname(out_path))

    combined = combine_resources(args.providers_dir, args.providers)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
