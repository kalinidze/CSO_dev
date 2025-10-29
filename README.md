## Cloud Resource Ontology (early stage)

This repository hosts early-stage work toward a cross-cloud resource ontology derived from semi-automatic analysis of Terraform provider schemas and documentation. The focus of the repo today is the data extraction, normalization, and supporting artefacts that will inform the ontology. The ontology itself is competed, but not yet finalized and is intentionally not the emphasis of this README.

### Project status
- This project is in active, early development.
- Structure, scripts, and file formats may change without notice.

## Repository structure

- `src/`
  - `combine_resources.py`: Aggregates provider resource definitions into a unified intermediate view for downstream processing.
  - `enrich_descriptions.py`: Adds and normalizes human-readable descriptions from provider documentation and other sources.
  - `embed_cluster_resources.py`: Produces vector embeddings for resources and clusters similar items to surface potential alignments.
  - `align_properties.py`: Attempts to align conceptually similar resource/property names across providers using signals from schemas, text, and embeddings.

- `providers/`
  - Provider-specific harvested inputs. Each provider folder (e.g., `aws/`, `azurerm/`, `google/`, `yandex/`) typically contains:
    - `schema.json`: Snapshot of Terraform provider resource schemas.
    - `docs_pages.jsonl`: Extracted documentation pages.
    - `index.json`: Lightweight index/lookup metadata supporting processing steps.
  - A top-level `docs_pages.jsonl` may also exist for shared or provider-agnostic docs.

- `artefacts/`
  - Outputs produced by the scripts in `src/`. Examples include:
    - `embedding_top_similar.json`: Nearest-neighbor results from resource embeddings.
    - `resource_top_similar.json`: Top similar resource pairs/groups discovered during clustering/alignment.
    - `enriched_descriptions.json`: Post-processed descriptions merged from multiple sources.
    - `providers_similar_temrs.json`: Candidate cross-provider term alignments produced during analysis.

- `README.md`
  - You are here.