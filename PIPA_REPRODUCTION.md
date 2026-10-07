# FusionAgent on PIPA

This branch adapts FusionAgent to the actual PIPA data geometry instead of
forcing the whole-body CCVID/MEVID protocol onto album photos.

## Design

PIPA has still images and a ground-truth **head** box. It does not contain gait
sequences, and many instances do not contain a reliable full body. The default
PIPA baseline therefore uses two tools:

- `pipa-head`: exact annotated head crop.
- `pipa-upper`: pose-agnostic upper-body crop derived from the head box.

The upper-body crop follows the PIPA region definition: the full-body rectangle
is 3x head width by 6x head height with the head at the top centre; upper body is
its upper half, i.e. 3x head width by 3x head height.

No relation, album, co-occurrence, OCR, VLM caption, or identity metadata is
given to FusionAgent. This keeps it a **dynamic expert-selection baseline** for
comparison with an evidence-aware method.

## What to train

1. **Cue experts (train first):** fine-tune separate ImageNet ResNet-50 models
   on PIPA `train` identities only, one for head and one for upper body.
   These weights are then frozen inside FusionAgent.
2. **FusionAgent policy:** LoRA/GRPO on Qwen2.5-VL-3B using the PIPA train
   identities and precomputed score matrices.
3. **Do not train on test0/test1 identities.** At evaluation, test0/test1 are
   person-specific registration/query folds. Run both directions and average.

The upstream CCVID FusionAgent LoRA checkpoint is useful only as a smoke-test or
initialisation experiment. It is not the primary PIPA result because the
available modalities and image distribution are different.

## Data preparation

The script accepts the metadata format from the maintained PIPA metadata
repository. Images must be in the same coordinate system as the supplied head
boxes.

```bash
export PIPA_META=/path/to/PIPA_dataset
export PIPA_IMAGES=/path/to/pipa/images
export PIPA_DATA_ROOT=/path/to/pipa_data_root
export PIPA_SPLIT=original   # original | album | time | day
bash src/scripts/fusionagent_pipa_prepare.sh
```

Outputs:

```text
<PIPA_DATA_ROOT>/PIPA_FusionAgent/
  manifest.csv
  pipa_head.h5
  pipa_upper.h5
```

Record the skipped-image count. Reconstructed Flickr copies can be missing or
have incompatible resolution; silently changing the benchmark population is
not acceptable.

## Train the visual experts

```bash
export PIPA_DATA_ROOT=/path/to/pipa_data_root
bash src/scripts/fusionagent_pipa_train_experts.sh
```

This creates:

```text
src/fusionagent/checkpoints/PIPA/pipa-head-resnet50.pth
src/fusionagent/checkpoints/PIPA/pipa-upper-resnet50.pth
```

The first reproducible baseline should freeze these experts during GRPO. Joint
expert+agent training is intentionally not implemented because it changes the
FusionAgent comparison and makes reward attribution unstable.

## Precompute centers and score matrices

```bash
bash src/scripts/fusionagent_pipa_precompute.sh
bash src/scripts/fusionagent_pipa_extract_scores.sh
```

Then edit `src/fusionagent/configs/train_config_test_pipa.yaml` and set
`root`. For the standard two-fold PIPA protocol:

- `pipa_protocol: 0to1`: gallery/register on test0, query on test1.
- `pipa_protocol: 1to0`: gallery/register on test1, query on test0.

Run both directions and average recognition accuracy. Do not report only the
easier direction.

## Train FusionAgent

```bash
bash src/scripts/fusionagent_pipa_grpo.sh
```

The trainable parameters in the main baseline are the Qwen LoRA parameters.
The two PIPA visual experts are frozen tools.

## Fair comparison with MRM + Evidence Agent

Use the same PIPA instances and the same head-derived image regions. FusionAgent
receives only pixels/crops and expert outputs. It must not receive relation
labels, album IDs, co-occurrence graphs, VLM captions, OCR, or temporal metadata
unless those sources are also enabled in the comparison method.

Recommended reporting:

- head-only expert
- upper-only expert
- static mean/Z-score fusion
- ACT fusion with both experts
- FusionAgent dynamic selection + ACT
- MRM visual-only
- MRM + Evidence Agent

Also report average number of expert calls per query. This is central to the
FusionAgent claim.

## Evaluation-path note

The upstream trainer initialises train-identity centers for tool execution, but
during evaluation it overwrites each tool's predicted identity from the
precomputed **test score matrix**. For PIPA, that score matrix is built from the
active test0/test1 query-gallery fold, so the identity shown back to the agent
comes from the correct PIPA gallery. The unused train-center computation is
still inefficient and should be removed in a later cleanup.

Run the complete pipeline once before publishing results and verify the
query/gallery counts and identity set for both directions.

No runtime tests or PIPA training have been executed by this commit.
