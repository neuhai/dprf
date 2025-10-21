# DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans

<p align="center">
  <em>Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2510.14205">
        <img src="https://img.shields.io/badge/arXiv-2510.14205-B31B1B.svg?style=plastic&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=plastic" alt="License: MIT">
    </a>
</p>

<p align="center">
  <img src="dprf.png" alt="DPRF Framework Overview" width="800"/>
</p>



## Overview

**DPRF (Dynamic Persona Refinement Framework)** is a generalizable framework that optimizes the alignment of LLM role-playing agents' behaviors with those of target individuals. The framework addresses the limitation of manually-created persona profiles by iteratively identifying the cognitive divergence, either through free-form or theory-grounded structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.

DPRF has been validated across five different LLMs and four diverse behavior-prediction scenarios:
*   **Formal Debates:** Predicting speaker statements on socio-political issues (targeting beliefs, intentions, and knowledge)
*   **Mental Health Expression:** Generating social media posts reflecting mental health states (targeting emotion)
*   **Opinionated Reviews:** Creating movie reviews consistent with emotional traits (targeting emotion and knowledge)
*   **Public Interviews:** Generating interview responses based on conversational context (targeting goals and intentions)

DPRF consistently improves behavioral alignment considerably over baseline personas and generalizes across models and scenarios. Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.


## Installation

1. Clone the repository:
```
git clone git@github.com:xxx/xxx.git
```

2. Set up environment:
```
uv sync
```

## Quick Start

### Basic Usage

Here's a simple example of using DPRF to refine a persona:
```
python Evaluation/debate/debate.py \
  --task_model_type sglang \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --refiner_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --refiner_model_type sglang \
  --output_dir "output" \
  --length 1 \
  --iterations 15 \
  --analysis_prompt_file prompts/analysis_brief.txt \
  --refinement_prompt_file prompts/refinement.txt \
  --instruction_prompt_file Evaluation/debate/prompts/instruction.txt \
  --initial_persona_file Evaluation/debate/prompts/initial_persona.txt \
  --data_dir Evaluation/debate/data/processed
```

All experimental scripts are available in the `scripts/` directory. Scripts prefixed with `claude_` are for Claude models (via AWS Bedrock), while `other_*` scripts are for open-source models (via sglang).

### Configuration

You can customize DPRF behavior through various command-line parameters:

1. Model Configuration
    - `--task_model`: Model name for task execution (e.g., `gpt-4o`, `meta-llama/Llama-3.2-3B-Instruct`)
    - `--task_model_type`: Task model type. Options: `openai`, `sglang`, `bedrock`
    - `--refiner_model`: Model name for persona refinement
    - `--refiner_model_type`: Refiner model type. Same options as task_model_type
    - `--openai_api_key`: OpenAI API key (defaults to environment variable `OPENAI_API_KEY`，required when using openai model)
    - `--bedrock_region`: AWS Bedrock region (e.g., `us-east-1`, required when using Bedrock)
    - `--model_kwargs_json`: JSON string or file path for model parameters (e.g., `'{"temperature": 0.7}'`)

2. DPRF Refinement Parameters:
    - `--iterations`: Maximum number of refinement iterations (default: 3). Controls how many rounds of persona optimization to perform. Higher values may improve alignment but increase runtime.
    - `--analysis_prompt_file`: Path to custom analysis prompt template (e.g., `prompts/analysis_brief.txt`, `prompts/analysis_structured.txt`)
    - `--refinement_prompt_file`: Path to custom refinement prompt template (e.g., `prompts/refinement.txt`)

3. Data and Output:
    - `--data_dir`: Path to data directory
    - `--output_dir`: Directory to save results (default: `results`)
    - `--length`: Number of examples to randomly select (default: use all data). Useful for quick testing.
    - `--initial_persona_file`: Path to initial persona template file
    - `--instruction_prompt_file`: Path to instruction prompt template file

## Citation

If you use DPRF in your research, please cite our paper:

```bibtex
@article{yao2025dprf,
  title={DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans},
  author={Yao, Bingsheng and Sun, Bo and Dong, Yuanzhe and Lu, Yuxuan and Wang, Dakuo},
  journal={arXiv preprint arXiv:2510.14205},
  year={2025}
}
```

Paper: [https://arxiv.org/abs/2510.14205](https://arxiv.org/abs/2510.14205)

## License

This project is licensed under the [MIT License](LICENSE).
