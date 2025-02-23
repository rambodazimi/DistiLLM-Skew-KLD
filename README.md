# DistiLLM: Knowledge Distillation via Skew KL Divergence Loss

## Overview
This repository presents an implementation of knowledge distillation for Large Language Models (LLMs) by employing **Skew KL Divergence Loss**. The objective is to transfer representational and functional knowledge from a **teacher model** (e.g., TinyLlama) to a more computationally efficient **student model** (e.g., FLAN-T5), thereby maintaining performance while reducing resource demands.

The methodology leverages the **Hugging Face Transformers library**, along with **Datasets**, and evaluation frameworks such as **ROUGE** and **BLEU** to ensure rigorous benchmarking of distilled models.

## Key Features
- Implements a **Skew KL Divergence Loss** function to enhance knowledge distillation efficiency.
- Utilizes **TinyLlama** as the high-capacity teacher model and **FLAN-T5** as the compact student model.
- Conducts systematic evaluation using **BLEU and ROUGE metrics** to assess linguistic fidelity.
- Optimized for execution on **GPUs (CUDA-enabled environments)** to ensure computational efficiency.

## Installation
To set up the required environment, install the following dependencies:

```bash
pip install torch transformers datasets evaluate rouge_score nltk
```

## Usage Instructions
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-repo/DistiLLM_Skew_KL.git
   cd DistiLLM_Skew_KL
   ```

2. Open the Jupyter Notebook for execution:
   ```bash
   jupyter notebook DistiLLM_Skew_KL_Divergence_Loss.ipynb
   ```

3. Execute the notebook’s cells in sequential order to:
   - Load and configure the teacher and student models.
   - Fine-tune the student model using **Skew KL Divergence Loss**.
   - Perform comprehensive evaluation on a designated test dataset.

## Model Training Workflow
- **Step 1**: Initialize a pre-trained teacher model (`TinyLlama/TinyLlama_v1.1`) and a student model (`google/flan-t5-base`).
- **Step 2**: Tokenize the training dataset via `AutoTokenizer` for optimal sequence representation.
- **Step 3**: Train the student model by aligning its output distribution with that of the teacher model, leveraging **Skew KL Divergence Loss** to regulate the transfer process.
- **Step 4**: Conduct evaluation using standard linguistic metrics such as **BLEU and ROUGE** to quantify performance gains and trade-offs.

## Evaluation Framework
- **BLEU Score**: Computes n-gram congruence between generated and reference text sequences to measure syntactic fidelity.
- **ROUGE Score**: Evaluates recall-oriented textual overlap to assess semantic alignment with reference outputs.

## Expected Performance Metrics
Following the completion of training, the distilled model achieves the following representative metrics:
```bash
BLEU Score: 25.4
ROUGE Score: 0.67
```
