# COS801_Group_Project_2025

Disaster Named Entity Recognition (NER) with DistilBERT & LoRA

Course: COS801 Project

Model: DistilBERT (Uncased) with Low-Rank Adaptation (LoRA)

Task: Token Classification / Named Entity Recognition

📌 Project Overview

This project implements a Named Entity Recognition (NER) system designed to extract specific disaster-related information from text. It identifies entities such as natural hazards, infrastructure damage, affected populations, and locations.

To achieve efficient fine-tuning on a limited compute budget, this project utilizes Parameter-Efficient Fine-Tuning (PEFT). Specifically, it employs LoRA (Low-Rank Adaptation) to inject trainable rank decomposition matrices into the distilbert-base-uncased model while freezing the pre-trained weights.

🏷️ Entity Labels

The model is trained to recognize the following entities:

Hazards: NaturalHazards, Floods, Fire

Impact: Death_And_Toll, AffectedPopulation, MissingPersons

Damage: InfrastructureDamage, CollapsedStructure, RoadBlocked, PowerOutage, WaterShortage

Context: Location, Date

🏗️ Model Architecture

Base Model: distilbert-base-uncased (6 Transformer layers).

Adaptation Technique: LoRA (Low-Rank Adaptation).

Rank (r): 8

Alpha: 32

Target Modules: q_lin (Query), v_lin (Value) attention layers.

Trainable Parameters: Only ~0.25% of the total parameters (approx. 170k out of 66M) are trained, significantly reducing memory usage.

📊 Dataset

The project uses a disaster dataset formatted in the CoNLL standard (token per line with corresponding label).

Total Samples: ~19,205 sentences

Split:

Train: 15,364

Validation: 1,920

Test: 1,921

⚙️ Installation & Requirements

To run this project, you need Python and the following libraries.

pip install torch numpy pandas scikit-learn
pip install transformers datasets evaluate seqeval peft accelerate bitsandbytes


🚀 Usage

The core logic is contained within the Jupyter Notebook.

Data Preparation: The notebook reads .txt files in CoNLL format and converts them into Hugging Face Dataset objects.

Tokenization: Uses AutoTokenizer (DistilBERT) with label alignment to handle sub-word tokenization.

Training: The Trainer API is used with the following hyperparameters:

Batch Size: 16

Learning Rate: 5e-5

Epochs: 3

Weight Decay: 0.01

Evaluation Strategy: Epoch-based

📈 Results

The model was evaluated on the test set using the seqeval metric.

Metric

Score

Accuracy

96.5%

F1-Score

68.9%

Recall

73.2%

Precision

65.1%

📂 File Structure

.
├── data/
│   └── dataset.txt        # CoNLL format data
├── notebooks/
│   └── disaster_ner_lora.ipynb
├── results/               # Training checkpoints
└── README.md


🧠 Why LoRA?

Instead of fine-tuning all 66 million parameters of DistilBERT, LoRA allows us to fine-tune only the injected adapter layers. This results in:

Faster Training: Fewer gradients to calculate.

Lower VRAM Usage: Capable of running on free-tier Colab GPUs (T4).

Modular Weights: The adapter weights are lightweight (~MBs) compared to the full model (~GBs).

📜 License

MIT License

