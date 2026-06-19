# SimEdit: Conditioning Matters — Stabilizing Inversion and Attention in Diffusion Image Editing

Official implementation of our ECML PKDD 2026 paper
**"Conditioning Matters: Stabilizing Inversion and Attention in Diffusion Image Editing"**.

SimEdit is a **training-free** framework for inversion-based diffusion image editing. It is built on the observation that the precision and structural alignment of textual conditioning strongly affect both inversion stability and cross-branch attention consistency. SimEdit has two complementary components:

- **Conditioning Refinement (CR):** expands the original source/target prompts with additional image-grounded details while preserving a shared semantic structure, stabilizing the diffusion velocity field and improving background preservation.
- **Token-wise Cross-Branch Attention Control (TCAC):** uses an LCS-based token alignment to separate *structure-preserving* and *edit-driving* tokens, and modulates their attention contributions asymmetrically during editing.

## Installation

```bash
# Python 3.10 recommended
pip install -r requires.txt
```

The implementation is based on `FLUX.1-dev` (via `diffusers`). A GPU with sufficient memory is required (we use NVIDIA A800; FLUX.1-dev needs about 48GB for 512x512 images).

## Quick Start

The full pipeline is in [`code/SimEdit.ipynb`](code/SimEdit.ipynb). Open it and run the cells in order. It will:

- (optionally) generate refined source/target prompts via an LLM,
- reconstruct the source image from inversion,
- produce the final edited result,
- save all intermediate outputs to folders.

### Conditioning Refinement (API key)

Conditioning refinement calls a vision-language model through the [OpenRouter](https://openrouter.ai/) API. Before running the refinement cell, set your own key:

```python
os.environ["OPENROUTER_API_KEY"] = "your-key-here"
```

This step is optional: you can also skip refinement and run editing with the prompts provided in [`mappings/Mapping_file_for_PIEBench.json`](mappings/Mapping_file_for_PIEBench.json), or supply your own refined prompts. The system prompt we use for refinement is in [`mappings/system_prompt.txt`](mappings/system_prompt.txt).

## Repository Contents

| Path | Description |
| --- | --- |
| `code/SimEdit.ipynb` | Complete SimEdit pipeline (conditioning refinement + token-wise cross-branch attention control), with reconstruction and editing outputs. |
| `code/calc_for_L_and_directional_deviation.py` | Estimation of the empirical Lipschitz constant `L` and the directional deviation used in the motivation analysis. |
| `code/src/model/` | Model code: diffuser pipeline, attention manipulation, and CLIP utilities. |
| `code/src/util/` | Utilities: token alignment (LCS), prompt running, metrics, and attention visualization. |
| `code/example.jpg` | Example input image used in the notebook. |
| `mappings/system_prompt.txt` | System prompt used for conditioning refinement; expands the original prompts with additional image-grounded details. |
| `mappings/Mapping_file_for_PIEBench.json` | Source/target prompt pairs used in our final experiments, generated with our system prompt and Gemini-2.5-Pro. |
| `mappings/CLIP*_mapping.json` | Source/target prompt pairs used to compute the CLIPSim* metric reported in the paper. |
| `supplementary/supplementary.pdf` | Supplementary material for the paper. |
| `requires.txt` | Required Python packages. |

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{zhan2026simedit,
  title={Conditioning Matters: Stabilizing Inversion and Attention in Diffusion Image Editing},
  author={Zhan, Zheyuan and Li, Hongchen and Wang, Can and Ma, Yinfei and Huang, Mingzhen and Bai, Ruoshi and Chen, Jiawei and Lyu, Siwei and Chen, Defang},
  journal={arXiv preprint arXiv:2606.14125},
  year={2026}
}
```
