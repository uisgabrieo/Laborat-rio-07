# Laboratório 07 - P2: Especialização de LLMs com LoRA e QLoRA

Pipeline completo de fine-tuning de um modelo de linguagem fundacional utilizando técnicas de eficiência de parâmetros (PEFT/LoRA) e quantização (QLoRA) para viabilizar o treinamento em hardwares com memória limitada.

## Estrutura do Projeto

```
lora&qlora/
├── lora_pipeline.py   # Classes do pipeline (DatasetGenerator, LoRAConfig, LoRAPipeline)
├── main.py            # Execução do pipeline completo
├── requirements.txt   # Dependências do projeto
├── .env               # Variáveis de ambiente (não versionado)
├── .gitignore
├── train.jsonl        # Dataset de treino gerado (90%)
└── test.jsonl         # Dataset de teste gerado (10%)
```

## Roteiro de Implementação

### Passo 1 — Engenharia de Dados Sintéticos

A classe `DatasetGenerator` usa a API do Llama para gerar pares de instrução/resposta no domínio de programação Python. O dataset é dividido em 90% treino e 10% teste e salvo no formato `.jsonl`.

### Passo 2 — Configuração da Quantização (QLoRA)

O `BitsAndBytesConfig` é configurado dentro do `LoRAPipeline` com:

- `load_in_4bit = True`
- `bnb_4bit_quant_type = "nf4"` (NormalFloat 4-bit)
- `bnb_4bit_compute_dtype = float16`

### Passo 3 — Arquitetura do LoRA

A classe `LoRAConfig` encapsula o `LoraConfig` da biblioteca `peft` com os hiperparâmetros obrigatórios:

| Hiperparâmetro | Valor | Descrição |
|---|---|---|
| `r` (rank) | 64 | Dimensão das matrizes de decomposição |
| `lora_alpha` | 16 | Fator de escala dos novos pesos |
| `lora_dropout` | 0.1 | Regularização para evitar overfitting |
| `task_type` | `CAUSAL_LM` | Modelos autoregressivos |

### Passo 4 — Pipeline de Treinamento

A classe `LoRAPipeline` orquestra o treinamento completo usando `SFTTrainer` da biblioteca `trl` com:

| Parâmetro | Valor |
|---|---|
| `optim` | `paged_adamw_32bit` |
| `lr_scheduler_type` | `cosine` |
| `warmup_ratio` | `0.03` |

Ao final, o adaptador é salvo com `trainer.model.save_pretrained()`.

## Instalação

```bash
pip install -r requirements.txt
```

> **Requisito:** GPU com suporte a CUDA e pelo menos 16 GB de VRAM (ex: Google Colab T4 ou A100).

## Configuração

Preencha o arquivo `.env` com sua chave da API:

```
LLAMA_API_KEY=chave
```

## Execução

```bash
python main.py
```

## Uso de IA Generativa

Utilizei ferramentas de IA generativa (Claude) como apoio para pesquisa e estudo dos conceitos envolvidos no laboratório — como o funcionamento do LoRA, QLoRA, quantização NF4 e o fluxo do SFTTrainer. Todo o código foi escrito e revisado por mim, Gabriel, com base no entendimento adquirido durante esse processo.

Partes consultadas/compreendidas com auxílio de IA, implementadas e revisadas por Gabriel.
