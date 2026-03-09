# Future Frontiers — Code Repository

**Book:** Future Frontiers: Harnessing AI, Safeguarding Privacy,
and Shaping Ethics in Pharma and Healthcare

**Author:** Jigar Sheth

## Structure
- `chapter01/` through `chapter21/` — one folder per chapter
- `shared/` — reusable utilities (data loaders, metrics, logging)
- `datasets/` — synthetic datasets (no real patient data ever)

## Quick Start
```bash
pip install -r requirements.txt
pytest chapter*/tests/ shared/tests/ -v
```

## License
Apache 2.0 — see LICENSE file.
