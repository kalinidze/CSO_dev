#!/usr/bin/env python3

import argparse
import json
import os
import sys
import re
import hashlib
from typing import Dict, Any, Optional, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_enriched_fingerprint_parts(
    name: str,
    description: Optional[str],
    enriched_prop: Optional[Dict[str, Any]],
    use_enriched: bool,
) -> List[str]:
    parts: List[str] = []
    parts.append(normalize_text(name))
    parts.append(normalize_text(description or ""))
    if use_enriched and enriched_prop:
        canonical_name = enriched_prop.get("canonical_name") or ""
        purpose = enriched_prop.get("purpose") or ""
        enriched_description = enriched_prop.get("enriched_description") or ""
        synonyms = enriched_prop.get("synonyms") or []
        parts.append(normalize_text(canonical_name))
        parts.append(normalize_text(purpose))
        parts.append(normalize_text(enriched_description))
        if isinstance(synonyms, list):
            # Join synonyms into one string to include as a single block
            syn_text = normalize_text(" ".join([str(s) for s in synonyms]))
            parts.append(syn_text)
        else:
            parts.append(normalize_text(str(synonyms)))
    # Filter empty
    return [p for p in parts if p]


def property_fingerprint_from_parts(parts: List[str]) -> str:
    h = hashlib.sha1()
    for idx, p in enumerate(parts):
        if idx:
            h.update(b"|")
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def align_properties(
    combined: Dict[str, Any],
    enriched: Optional[Dict[str, Any]],
    use_enriched: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    id_to_props: Dict[str, List[Dict[str, Any]]] = {}

    # Annotated copy of combined (resources -> properties -> add id)
    annotated: Dict[str, Any] = {}

    for resource_type, res in combined.items():
        res_desc = res.get("description")
        props = res.get("properties", {}) or {}
        enriched_resource = enriched.get(resource_type, {}) if enriched else {}
        enriched_props_map: Dict[str, Any] = enriched_resource.get("properties", {}) if isinstance(enriched_resource, dict) else {}

        new_props: Dict[str, Any] = {}
        for prop_path, meta in props.items():
            name = prop_path.split(".")[-1]
            desc = meta.get("description") or ""
            enriched_prop = enriched_props_map.get(prop_path) if isinstance(enriched_props_map, dict) else None

            parts = build_enriched_fingerprint_parts(name, desc, enriched_prop, use_enriched)
            fid = property_fingerprint_from_parts(parts)

            # Append to index list
            item = {
                "resource": resource_type,
                "path": prop_path,
                "name": name,
                "description": desc,
            }
            if enriched_prop and use_enriched:
                item["canonical_name"] = enriched_prop.get("canonical_name")
                item["purpose"] = enriched_prop.get("purpose")
                item["synonyms"] = enriched_prop.get("synonyms")
                item["enriched_description"] = enriched_prop.get("enriched_description")
            id_to_props.setdefault(fid, []).append(item)

            # Add id to property meta (and optionally attach enriched hints for downstream use)
            enriched_meta = dict(meta)
            enriched_meta["id"] = fid
            if enriched_prop and use_enriched:
                enriched_meta["canonical_name"] = enriched_prop.get("canonical_name")
                enriched_meta["purpose"] = enriched_prop.get("purpose")
                enriched_meta["synonyms"] = enriched_prop.get("synonyms")
                enriched_meta["enriched_description"] = enriched_prop.get("enriched_description")
            new_props[prop_path] = enriched_meta
        # Copy resource with annotated properties
        annotated[resource_type] = {
            "description": res_desc,
            "properties": new_props,
        }

    return id_to_props, annotated


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align semantically similar properties by name/description and enriched fields, then assign IDs.",
    )
    parser.add_argument(
        "--providers-dir",
        default=os.path.join(os.getcwd(), "providers"),
        help="Directory containing combined_resources.json (default: ./providers)",
    )
    parser.add_argument(
        "--combined",
        default=None,
        help="Path to combined_resources.json (default: <providers-dir>/combined_resources.json)",
    )
    parser.add_argument(
        "--enriched",
        default=None,
        help="Path to enriched_descriptions.json (default: <providers-dir>/enriched_descriptions.json)",
    )
    parser.add_argument(
        "--no-enriched",
        action="store_true",
        help="Ignore enriched fields and use only name+description",
    )
    parser.add_argument(
        "--properties-out",
        default=None,
        help="Output properties index JSON (default: <providers-dir>/properties.json)",
    )
    parser.add_argument(
        "--annotated-out",
        default=None,
        help="Output annotated combined resources JSON (default: <providers-dir>/combined_resources_annotated.json)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    combined_path = args.combined or os.path.join(args.providers_dir, "combined_resources.json")
    enriched_path = args.enriched or os.path.join(args.providers_dir, "enriched_descriptions.json")
    props_out = args.properties_out or os.path.join(args.providers_dir, "properties.json")
    annotated_out = args.annotated_out or os.path.join(args.providers_dir, "combined_resources_annotated.json")

    if not os.path.exists(combined_path):
        print(f"Missing combined resources: {combined_path}", file=sys.stderr)
        return 2

    combined = read_json(combined_path)
    enriched: Optional[Dict[str, Any]] = None
    use_enriched = not args.no_enriched
    if use_enriched and os.path.exists(enriched_path):
        enriched = read_json(enriched_path)
    else:
        enriched = None

    id_to_props, annotated = align_properties(combined, enriched, use_enriched)

    write_json(props_out, id_to_props)
    write_json(annotated_out, annotated)

    print(f"Wrote {props_out}")
    print(f"Wrote {annotated_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
