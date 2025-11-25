# Differential Privacy with Fine-tuning GPT-2 (Multi-Agent Clinical Notes)

## Overview
This project demonstrates how to fine-tune GPT-2 models for clinical note generation using both Differential Privacy (DP) and non-DP (baseline) approaches. It leverages LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning and includes multi-agent pipelines for automated, privacy-aware clinical documentation. The project is designed for healthcare data, focusing on protecting sensitive information while maintaining model utility.

## Features
- **Differential Privacy Fine-tuning:** Train GPT-2 models with Opacus to ensure privacy guarantees on clinical data.
- **Non-DP Baseline:** Train comparable models without privacy constraints for side-by-side evaluation.
- **LoRA Adapters:** Efficient parameter tuning for large language models.
- **Multi-Agent Inference:** Automated agents for intake, SOAP note generation, coding, and safety auditing.
- **PHI Redaction:** Built-in scrubbing of personal health information from both input and output.
- **Privacy Evaluation:** Membership inference attacks (MIA) to assess privacy leakage.
- **Comprehensive Metrics:** Training, validation, and privacy metrics are logged and visualized.

## Directory Structure
```
Differential Privacy/
├── Agent_With_DP.py           # Inference agent for DP model
├── Agent_Without_DP.py        # Inference agent for non-DP model
├── FineTuning_With_DP.py      # Training script with DP
├── FineTuning_Without_DP.py   # Training script without DP
├── requirements.txt           # Python dependencies
└── ...                        # Other scripts, data, and outputs
```

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/SyedIbad123/Differential-Privacy-with-Finetuning-GPT-2-including-multi-agent.git
   cd Differential-Privacy-with-Finetuning-GPT-2-including-multi-agent
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Prepare your data:**
   - Place your clinical CSV (e.g., `aci_bench_refined.csv`) in the working directory.
   - Update the `CSV_PATH` in the config section of the training scripts if needed.

## Usage
### 1. Fine-tuning with Differential Privacy
Run the DP training script:
```sh
python FineTuning_With_DP.py
```
- Model checkpoints and logs will be saved in `runs/dp_lora_<timestamp>/`.

### 2. Fine-tuning without Differential Privacy (Baseline)
Run the baseline training script:
```sh
python FineTuning_Without_DP.py
```
- Model checkpoints and logs will be saved in `runs/lora_baseline_<timestamp>/`.

### 3. Inference with Multi-Agent Pipeline
- For DP model:
  ```sh
  python Agent_With_DP.py
  ```
- For non-DP model:
  ```sh
  python Agent_Without_DP.py
  ```

### 4. Privacy Evaluation
- Membership inference attacks and privacy metrics are computed automatically after training.
- Results and plots are saved in the corresponding `runs/` directory.

## Key Components
- **Agent_With_DP.py / Agent_Without_DP.py:**
  - Multi-agent workflow for clinical note generation, coding, and safety auditing.
  - PHI redaction and output sanitization.
- **FineTuning_With_DP.py / FineTuning_Without_DP.py:**
  - Data loading, augmentation, and preprocessing.
  - LoRA-based fine-tuning (with/without DP).
  - Early stopping, logging, and privacy evaluation.
- **requirements.txt:**
  - All necessary Python packages for training and inference.

## Notes
- The project is designed for research and educational purposes. Use with real clinical data requires careful review of privacy and compliance requirements.
- GPU is recommended for training.
- For best results, adjust hyperparameters and data paths as needed.

## References
- [Opacus (PyTorch Differential Privacy)](https://opacus.ai/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## License
This project is licensed under the MIT License.
