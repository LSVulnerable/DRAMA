# DRAMA: Next-Gen Dynamic Orchestration for Resilient Multi-Agent Ecosystems in Flux

Code release for the paper:

**Xinkui Zhao, Yifan Zhang, Sai Liu, Naibo Wang, Guanjie Cheng, Yueshen Xu, Chang Liu, Shuiguang Deng, Jianwei Yin**  
**DRAMA: Next-Gen Dynamic Orchestration for Resilient Multi-Agent Ecosystems in Flux**

DRAMA is a dynamic orchestration framework for embodied multi-agent systems in VirtualHome-Social. It supports event-triggered rescheduling, fault-tolerant task takeover, and collective spatial reasoning for teams that must keep working under agent arrivals, dropouts, and recoveries.

[Paper](https://arxiv.org/abs/2508.04332)

## Overview

<img src="assets/overview.png" alt="DRAMA overview" width="100%">

DRAMA uses a three-layer design:

- Strategic Layer: affinity-driven scheduling and trust-chain task takeover.
- Collective Intelligence Layer: merges distributed observations into shared spatial priors.
- Autonomous Layer: lets each agent act, monitor, and adapt locally.

<img src="assets/strategiclayer.png" alt="DRAMA strategic layer" width="100%">

<img src="assets/casestudy.png" alt="DRAMA case study" width="100%">

## Highlights

- Dynamic agent-task allocation with Hungarian matching over affinity scores.
- Hierarchical trust chains for resilient takeover when agents fail or drop out.
- Shared perception and predictive reasoning from distributed observations.
- Evaluation in embodied VirtualHome-Social under static and dynamic populations.

## Repository Structure

- `envs/cwah/algos/arena_mp2.py`: environment-agent interface.
- `envs/cwah/algos/comms.py`: communication module.
- `envs/cwah/agents/`: LLM agent implementation.
- `envs/cwah/testing_agents/test_drama.py`: DRAMA evaluation entry point.
- `envs/cwah/scripts/run_drama.sh`: release run script.
- `envs/cwah/LLM/`: DRAMA prompt templates and LLM configuration helpers.
- `envs/cwah/dataset/`: evaluation datasets.

## Installation

1. Clone the dependencies expected by this repo:

```bash
git clone --branch wah https://github.com/xavierpuigf/virtualhome.git envs/virtualhome
```

2. Download the Linux x86-64 simulator executable from the VirtualHome release page and place it under:

```text
envs/executable/
```

3. Create the Python environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The project pins Python 3.10 in `.python-version`. A legacy Conda environment file is also kept at `drama.yml` for users who prefer Conda.

## Configuration

Edit `envs/cwah/scripts/llm_configs.json` with your API key or local model endpoint. For example:

```json
{
  "gpt-4.1": {
    "api_key": "API_KEY",
    "model": "gpt-4.1"
  }
}
```

Organization instructions are stored in `envs/cwah/testing_agents/organization_instructions.csv`.

Example:

```csv
code,instruction
3,Agent 1 is the leader to coordinate the task.
```

## Run

Use the release script to run DRAMA:

```bash
cd envs/cwah/scripts/
uv run bash run_drama.sh
```

Logs are written under `envs/cwah/log/`, and detailed episode records are written under `envs/cwah/test_results/`.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{zhao2026drama,
  title={DRAMA: Next-Gen Dynamic Orchestration for Resilient Multi-Agent Ecosystems in Flux},
  author={Zhao, Xinkui and Zhang, Yifan and Liu, Sai and Wang, Naibo and Cheng, Guanjie and Xu, Yueshen and Liu, Chang and Deng, Shuiguang and Yin, Jianwei},
  year={2026},
  note={Code release}
}
```

## Acknowledgements

This project builds on [VirtualHome](https://github.com/xavierpuigf/virtualhome), [AutoGen](https://github.com/microsoft/autogen), and prior embodied multi-agent work.
