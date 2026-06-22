# Text-to-Cypher Fine-tuning

Fine-tuning `HuggingFaceTB/SmolLM2-135M-Instruct` on the `RomanTeucher/text2cypher-curated` dataset to generate Cypher queries from natural language questions and graph schemas.

---

## Setup

### Requirements
- Python 3.10+
- CPU is sufficient (no GPU required)

### Installation
```bash
git clone https://github.com/sejal-0502/text2cypher.git
cd text2cypher
pip install -r requirements.txt
```

---

## Reproduce Results

### 1. Baseline Evaluation (no fine-tuning)
```bash
mkdir results
python src/evaluate.py --model HuggingFaceTB/SmolLM2-135M-Instruct --output results/baseline_results.json
```

### 2. Fine-tuning
```bash
# 1 epoch
python src/train.py --output_dir checkpoints --num_epochs 1 --batch_size 4 --learning_rate 5e-5

# 3 epochs
python src/train.py --output_dir checkpoints_3epoch --num_epochs 3 --batch_size 4 --learning_rate 5e-5
```

### 3. Evaluate Fine-tuned Models
```bash
# 1 epoch
python src/evaluate.py --model checkpoints/final --output results/finetuned_1epoch_results.json

# 3 epochs
python src/evaluate.py --model checkpoints_3epoch/final --output results/finetuned_3epoch_results.json
```

### 4. Evaluate directly from HuggingFace Hub
If you don't want to train locally, you can evaluate directly using our uploaded checkpoints:

```bash
# 1 epoch model from HuggingFace Hub
python src/evaluate.py --model SejalMutakekar/text2cypher-smollm2-135m/model_small --output results/hub_model_results.json

# 3 epoch model from HuggingFace Hub
python src/evaluate.py --model SejalMutakekar/text2cypher-smollm2-135m/model_large --output results/hub_3epoch_results.json
```

**Model checkpoints are available at:**
[https://huggingface.co/SejalMutakekar/text2cypher-smollm2-135m](https://huggingface.co/SejalMutakekar/text2cypher-smollm2-135m)

- `model_small/` — fine-tuned for 1 epoch (Exact Match: 0.22, BLEU: 0.4109)
- `model_large/` — fine-tuned for 3 epochs (Exact Match: 0.40, BLEU: 0.5885)

---

## Results

| Setting | Exact Match | BLEU Score |
|---------|-------------|------------|
| Baseline (no fine-tuning) | 0.00 | 0.0143 |
| Fine-tuned (1 epoch) | 0.22 | 0.4109 |
| Fine-tuned (3 epochs)| 0.40 | 0.5885 |

---

## Design Decisions

### Model
We use `SmolLM2-135M-Instruct` — a small 135M parameter instruction-tuned model. 
It is small enough to fine-tune on CPU in reasonable time while still being 
capable enough to learn Cypher patterns from the training data.

### Prompt Format
We use the model's native chat template with three roles:
- `system` — instructs the model to generate Cypher queries
- `user` — contains the schema and natural language question
- `assistant` — contains the target Cypher query (training) or is left empty (evaluation)

During training, prompt tokens are masked with `-100` so the loss is computed 
only on the Cypher tokens.

### Dataset
We use `RomanTeucher/text2cypher-curated` which provides 1000 train, 75 val, 
and 50 test examples. Each example contains a natural language question, a graph 
schema, and a ground truth Cypher query.

### Evaluation Metrics
We use two metrics:

**1. Exact Match** : It checks whether the generated Cypher is character-for-character 
identical to the ground truth. This is strict but unambiguous — a score of 0.22 
means 11 out of 50 queries were perfectly correct.

**2. BLEU Score** : It measures token-level overlap between the generated and ground 
truth query. Gives partial credit for queries that are structurally close but not 
identical. A score of 0.41 indicates the model has learned the general structure 
of Cypher even when it doesn't get every detail right.

**What these metrics miss:**
- **Semantic equivalence** : Two different Cypher queries can return identical 
results. Both exact match and BLEU would penalize a correct query that is 
phrased differently from the ground truth.
- **Execution correctness** : The only true measure would be running both queries 
against a live Neo4j database and comparing results. This requires a running 
Neo4j instance and is out of scope here.

### Training
- 1 epoch takes ~24 minutes on CPU
- Loss dropped from 1.398 → 0.224 during 1 epoch of training
- Batch size of 4 and learning rate of 5e-5 are standard fine-tuning defaults
- We use `DataCollatorForSeq2Seq` for dynamic padding within batches

---

## Limitations

- **Model size** — at 135M parameters, the model struggles with complex multi-hop 
queries, aggregations, and queries requiring multiple WITH clauses. It performs 
best on simple MATCH/WHERE/RETURN patterns.
- **Dataset inconsistency** — the schema format varies significantly across examples 
(some use `Graph schema: Relevant node labels...`, others use `Node properties: **Label**...`). This inconsistency makes it harder for the model to learn a unified representation.
- **Data quality** — some examples have type mismatches between the question and the ground truth Cypher (e.g.treating an INTEGER as a STRING in the query). The model learns these inconsistencies as if they were correct.
- **Exact match is harsh** — the model may generate semantically correct Cypher that differs in whitespace, alias names, or property order, all of which would score 0 on exact match.
- **No execution validation** — we cannot verify whether generated queries are actually executable or return correct results without a live Neo4j instance.
- **Overfitting risk** — with only 1000 training examples and a model that can memorize patterns quickly, there is a risk of overfitting, especially beyond 1-2 epochs. 