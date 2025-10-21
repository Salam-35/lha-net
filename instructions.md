Title: Make LHA‑Net Modality‑Aware (CT + MRI)

Goal
- Allow training and inference on mixed CT and MRI data without manual intervention.
- Auto‑detect modality per case and apply the correct preprocessing.
- Preserve current CT behavior as the default so existing CT‑only workflows remain unchanged.

Why It Fails Today
- Preprocessing assumes CT (HU):
  - Global clip_range [-200, 300] and body mask threshold around -500 are CT‑specific.
  - MRI intensities are arbitrary (not HU); clipping to CT ranges destroys MRI signal.
- Inference repeats CT‑style clipping/normalization.
- No modality detection routed through the dataset or predictor.

High‑Level Plan
1) Add robust modality detection (CT vs MRI) per image.
2) Make preprocessing modality‑aware (clipping, masking, normalization).
3) Plumb modality through dataset samples and saved metadata.
4) Make inference preprocessing modality‑aware as well.
5) Update config to support per‑modality parameters while keeping defaults.
6) Verify with the existing quick and full data checks + a loader test on mixed data.

Code Changes (surgical and contained)
1) src/data/preprocessing.py
   - Add detect_modality(image: np.ndarray, path: Path, header/meta) -> Literal['CT','MRI']
     Heuristics (combined for robustness):
       - If a significant fraction of voxels < -200 or min < -500 => CT.
       - If min < 0 and max > 1000 => CT (typical HU span).
       - Else => MRI.
       - Optional: token hints from file/folder names containing 'ct'/'mr'.

   - Add MRI‑specific body mask creation:
       - For MRI, use Otsu threshold (on a smoothed image), keep the largest connected component, then morphological close/fill.
       - Keep existing CT mask: threshold at about -500 HU + clean up.

   - Make intensity normalization conditional:
       - CT: keep current path (clip to CT window e.g., [-200, 300], then z‑score within body mask).
       - MRI: DO NOT use CT clipping. Use robust normalization, e.g., percentile [p1, p99] clipping then z‑score/median‑IQR within the MRI body mask.

   - Save modality in returned metadata and in on‑disk metadata (when save_preprocessed=True).

   - Optionally expose target_spacing overrides per modality (default stays [1.5, 1.5, 1.5]).

2) src/data/preprocessing.py (AMOSDataset)
   - Store case modality in case_info (from preprocessed metadata if available, else re‑detect on load).
   - Return modality in __getitem__ alongside 'image', 'label', 'case_id'. Training code won’t depend on it, but it’s useful for debugging/analysis.

3) inference.py (LHANetPredictor)
   - Add a small detect_modality in preprocess path (reuse logic — consider moving to a shared util).
   - Branch preprocessing by modality:
       - CT: current clip + z‑score.
       - MRI: percentile clip + robust z‑score.
   - Add optional override flag (e.g., --force-modality {ct,mri}) for deterministic runs if needed.

4) Config: configs/lha_net_config.yaml
   - Keep existing keys for backward compatibility.
   - Add a nested structure with safe defaults:
     data:
       preprocessing:
         modality: auto  # auto|ct|mri
         ct:
           clip_range: [-200, 300]
           body_mask_threshold: -500
           normalization: z_score
         mri:
           normalization: percentile  # robust
           percentiles: [1.0, 99.0]
           body_mask: otsu
     - If the nested block isn’t present, fall back to current behavior (CT).

Verification Steps
1) Quick sanity (flags mixed modalities and CT‑clipping risk)
   - Run: `python3 quick_data_check.py`
   - Expect: If both CT and MRI are present, it warns. If MRI exists with CT clip_range, it flags a critical issue.

2) Deeper scan
   - Run: `python3 check_data.py`
   - Expect: Summary of CT vs MRI counts, sample ranges, and label checks. This confirms detection heuristics match your data.

3) Loader validation on mixed data
   - After implementing changes, run: `python3 test_data_loading.py`
   - Expect: No shape/range issues. Mean around ~0, std reasonable for both modalities. No constant images. Unique label values present.

4) Training smoke test
   - With a small subset (e.g., 4–8 cases with both modalities), run 1–2 epochs using the existing `train.py`.
   - Expect: No runtime errors, loss decreases, memory within limits. Dice calculation should run for validation.

5) Inference smoke test
   - Run `inference.py` on one CT and one MRI file. Confirm preprocessing logs show different modality‑specific logic and outputs look reasonable (visually or via organ voxel counts).

Design Notes and Heuristics
- Detection is heuristic by necessity for NIfTI; DICOM has explicit Modality but AMOS22 is NIfTI.
- Combining conditions (min < -500 OR a noticeable fraction of voxels below -200 OR broad negative–positive span) is reliable for CT in AMOS‑style data.
- MRI occasionally has small negative values (preprocessing history). Use fractions/percentiles rather than just min.
- Store `modality` in metadata to avoid re‑detecting downstream; still re‑detect as fallback if metadata is missing.

Risks and Edge Cases
- Pre‑scaled CT without negative values could appear MRI‑like. Filename hints and span (max > ~2000) help.
- MRI sequences vary; percentile ranges [1,99] are robust, but keep configurable.
- If images are pre‑cropped tightly, CT fraction below -200 may be tiny; rely on the span rule.
- Keep an override parameter to force a modality in case of mis‑detection.

Rollout Plan
1) Implement detection and modality‑aware preprocessing in src/data/preprocessing.py.
2) Add modality propagation in AMOSDataset and metadata saving.
3) Update inference preprocessing with detection/override.
4) Extend config to include optional per‑modality blocks (defaults preserved).
5) Re‑run quick_data_check.py and check_data.py to confirm detection and that no MRI would be clipped incorrectly.
6) Run test_data_loading.py; then a short training and two inference samples (one CT, one MRI).

Can It Work? How To Verify Now
- The repo already includes `quick_data_check.py` and `check_data.py` that detect mixed modalities and warn about CT‑only clipping. Run them first to confirm your dataset is mixed and to quantify risk.
- If your environment lacks dependencies (e.g., nibabel), install from requirements and re‑run the checks.
- Given the current code structure (single central preprocessor + dataset + inference), the changes are localized and low‑risk. CT‑only users will see no change; mixed CT/MRI will get the right normalization per‑case.

Next Steps (once approved)
- I’ll implement the detection utilities, wire them into preprocessing and inference, adjust the config, and provide a short verification log with the existing tests.

