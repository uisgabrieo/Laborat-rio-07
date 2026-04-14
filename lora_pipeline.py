import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq


class DatasetGenerator:

    SYSTEM_PROMPT = (
        "Você é um especialista em {domain}. "
        "Gere perguntas práticas e objetivas sobre o tema, "
        "junto com respostas detalhadas e corretas."
    )

    def __init__(self, api_key: str, domain: str = "programação Python", output_dir: str = "."):
        self.client = Groq(api_key=api_key)
        self.domain = domain
        self.output_dir = output_dir

    def _generate_single_pair(self) -> dict:
        system = self.SYSTEM_PROMPT.format(domain=self.domain)
        user_msg = (
            f"Gere UM único par de instrução e resposta sobre {self.domain}. "
            "Responda APENAS em JSON com as chaves 'prompt' e 'response', "
            "sem nenhum texto adicional fora do JSON."
        )

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.9,
        )

        raw = completion.choices[0].message.content.strip()
        pair = json.loads(raw)

        if "prompt" not in pair or "response" not in pair:
            raise ValueError(f"Par inválido retornado pela API: {raw}")

        return {"prompt": pair["prompt"], "response": pair["response"]}

    def generate_pairs(self, n: int = 50) -> list[dict]:
        pairs: list[dict] = []
        print(f"[DatasetGenerator] Gerando {n} pares para o domínio: '{self.domain}'")

        for i in range(n):
            pair = self._generate_single_pair()
            pairs.append(pair)
            print(f"  [{i + 1}/{n}] par gerado.")

        return pairs

    def split_and_save(
        self,
        pairs: list[dict],
        train_ratio: float = 0.9,
        train_filename: str = "train.jsonl",
        test_filename: str = "test.jsonl",
    ) -> tuple[str, str]:
        random.shuffle(pairs)

        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        test_pairs = pairs[split_idx:]

        train_path = os.path.join(self.output_dir, train_filename)
        test_path = os.path.join(self.output_dir, test_filename)

        self._write_jsonl(train_pairs, train_path)
        self._write_jsonl(test_pairs, test_path)

        print(
            f"[DatasetGenerator] Dataset salvo:\n"
            f"  Treino : {train_path} ({len(train_pairs)} exemplos)\n"
            f"  Teste  : {test_path}  ({len(test_pairs)} exemplos)"
        )

        return train_path, test_path

    @staticmethod
    def _write_jsonl(records: list[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class LoRAConfig:

    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    task_type: str = "CAUSAL_LM"

    def build(self):
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            target_modules=self.target_modules,
            task_type=TaskType.CAUSAL_LM,
        )


class LoRAPipeline:

    def __init__(
        self,
        model_name: str,
        train_dataset_path: str,
        lora_cfg: Optional[LoRAConfig] = None,
        output_dir: str = "lora_adapter",
    ):
        self.model_name = model_name
        self.train_dataset_path = train_dataset_path
        self.lora_cfg = lora_cfg or LoRAConfig()
        self.output_dir = output_dir

        self._model = None
        self._tokenizer = None
        self._trainer = None

    def _load_model_and_tokenizer(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(f"[LoRAPipeline] Carregando modelo: {self.model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self._model.config.use_cache = False

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

    def _apply_lora(self):
        from peft import get_peft_model, prepare_model_for_kbit_training

        self._model = prepare_model_for_kbit_training(self._model)
        self._model = get_peft_model(self._model, self.lora_cfg.build())

    def _load_dataset(self):
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=self.train_dataset_path, split="train")

        def format_instruction(example: dict) -> dict:
            return {
                "text": (
                    f"### Instrução:\n{example['prompt']}\n\n"
                    f"### Resposta:\n{example['response']}"
                )
            }

        return dataset.map(format_instruction)

    def train(self) -> None:
        from transformers import TrainingArguments
        from trl import SFTTrainer

        self._load_model_and_tokenizer()
        self._apply_lora()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            fp16=True,
            report_to="none",
        )

        self._trainer = SFTTrainer(
            model=self._model,
            train_dataset=self._load_dataset(),
            args=training_args,
            tokenizer=self._tokenizer,
            dataset_text_field="text",
            max_seq_length=512,
            packing=False,
        )

        print("[LoRAPipeline] Iniciando treinamento...")
        self._trainer.train()
        print("[LoRAPipeline] Treinamento concluído.")

    def save(self) -> str:
        if self._trainer is None:
            raise RuntimeError("Nenhum treinamento foi executado. Chame train() primeiro.")

        os.makedirs(self.output_dir, exist_ok=True)
        self._trainer.model.save_pretrained(self.output_dir)
        self._tokenizer.save_pretrained(self.output_dir)

        print(f"[LoRAPipeline] Adaptador salvo em: {self.output_dir}")
        return self.output_dir
