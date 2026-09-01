# TMC-LM Obsidian Vault

> This folder `tmc-llm` is now an Obsidian vault. Open it via Obsidian: **Open folder as vault**.

## Structure
```
tmc-llm/
  documents.md              <- project reference (Chapter 1-12)
  data/raw/tmc_sources/     <- OFFICIAL TMC knowledge source (training data)
  data/processed/           <- generated dataset (auto, do not edit)
  configs/train_lora.yaml   <- training config
  00 - Vault Guide.md       <- this file
  attachments/              <- images/pdfs na i-drag mo sa notes
  templates/                <- Obsidian templates
```

## Workflow: Paano magdagdag ng knowledge?

1.  Gumawa ng note sa `data/raw/tmc_sources/` - e.g. `Vision-Mission.md`, `Student-Handbook-2026.md`
2.  Gamitin ang template: `templates/TMC Source.md`
3.  Ilagay ang official content (vision, mission, programs, policies)
4.  Run sa PowerShell:
    ```powershell
    .\scripts\prepare_dataset.ps1
    .\scripts\train_lora.ps1
    ```

> Lahat ng `.md`, `.txt`, `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json` sa `data/raw/tmc_sources/` ay auto na nagiging training data via `src/tmc_llm/document_loader.py:12` at `src/tmc_llm/dataset_builder.py:210`.

## Tips sa Obsidian

-  **Graph View**: makikita mo links ng TMC knowledge
-  **Tags**: gumamit ng `#tmc/vision`, `#tmc/program`, `#tmc/policy`
-  **Links**: `[[Vision-Mission]]` para mag-connect ng notes
-  **Ignored folders**: `.venv`, `models`, `external`, `data/processed` - hindi mo kailangan buksan sa Obsidian (naka-exclude sa search kung gusto mo i-config sa Settings > Files & Links > Excluded files)

## Buksan sa Obsidian

1. Open Obsidian
2. `Open folder as vault` -> piliin `C:\Users\Argie Cabudbud\tmc-llm`
3. Done - makikita mo na lahat ng notes

## Validation
- Test vault: `Test-Path ".obsidian/app.json"` => True
- Check loader: `python -m tmc_llm.dataset_builder --help`
