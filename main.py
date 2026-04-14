import os
import pathlib

# Fix encoding do Windows para bibliotecas que leem arquivos internos (trl, jinja)
_orig_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)
pathlib.Path.read_text = _utf8_read_text

from dotenv import load_dotenv

from lora_pipeline import DatasetGenerator, LoRAConfig, LoRAPipeline

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = "lora_adapter"


def main() -> None:
    print("=" * 60)
    print("  Lab 07 – LoRA / QLoRA Fine-tuning Pipeline")
    print("=" * 60)

    if not GROQ_API_KEY:
        raise EnvironmentError("Defina GROQ_API_KEY no arquivo .env antes de executar.")

    print("\n[Passo 1] Gerando dataset sintético...")
    generator = DatasetGenerator(api_key=GROQ_API_KEY, domain="programação Python")
    pairs = generator.generate_pairs(n=50)
    train_path, _ = generator.split_and_save(pairs, train_ratio=0.9)

    print("\n[Passos 2-4] Configurando e treinando...")
    pipeline = LoRAPipeline(
        model_name=MODEL_NAME,
        train_dataset_path=train_path,
        lora_cfg=LoRAConfig(r=64, lora_alpha=16, lora_dropout=0.1),
        output_dir=OUTPUT_DIR,
    )

    pipeline.train()
    pipeline.save()

    print("\n[Concluído] Adaptador LoRA disponível em:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
