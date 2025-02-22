import os
from functools import partial
from pathlib import Path

import torch
import torchao
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TorchAoConfig,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

torchao.quantization.utils.recommended_inductor_config_setter()

os.environ["WANDB_PROJECT"] = "Zeta_Qwen_CodeMonk"

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Found CUDA accelerator...")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Found MPS accelerator...")
else:
    DEVICE = "cpu"
    print("No accelerator found. Defaulting to cpu..")

wandb_run_name = "torch_qwen0.5b_sft_v1"


# define training params
MAX_SEQ_LENGTH = 2048
RANK = 32
ALPHA = RANK
LORA_DROPOUT = 0.05
RANDO_SEED = 42
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
NUM_EPOCHS = 3
GRAD_ACCM = 2
WARM_UP_RATIO = 0.1
LR_SCHEDULER = "cosine"
OUTPUT_DIR = os.path.join(Path.home(), "Data/Models/ZetaQwen_0dot5b")

ALPACA_PROMPT = """### Instruction:
You are a code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location.

### User Edits:

{}

### User Excerpt:

{}

### Response:

{}
"""

RESPONSE_TEMPLATE = "### Response:\n\n"

EOS_TOKEN = None  # add when tokenizer is loaded
ORIGINAL_START_MARKER = "<|editable_region_start|>"
ORIGINAL_END_MARKER = "<|editable_region_end|>"


def format_example(events, input, output):
    return ALPACA_PROMPT.format(events, input, output)


def formatting_prompts_func(examples):
    events = examples["events"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for events, input, output in zip(events, inputs, outputs):
        output_start_index = output.find(ORIGINAL_START_MARKER)
        output_focused_region = output[output_start_index:]
        output = output_focused_region

        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = format_example(events, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


def filter_long_sequences(tokenizer, examples):
    tokenized = tokenizer(examples["text"])
    return len(tokenized["input_ids"]) <= 1500


def load_zeta_dataset(tokenizer, num_samples: int = -1):
    """load zeta dataset and return num_samples"""
    seed = 42
    revision = "5920488"
    dataset = load_dataset("zed-industries/zeta", revision=revision)

    # load subset of samples
    if num_samples != -1:
        train_dataset = dataset["train"].shuffle(seed=seed).select(range(num_samples))
        eval_dataset = dataset["eval"].shuffle(seed=seed).select(range(32))
        dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})

    # put data elements in alpaca format
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # apply filtering for large length samples
    filter_func = partial(filter_long_sequences, tokenizer)
    train_dataset = dataset["train"].filter(filter_func)
    eval_dataset = dataset["eval"].filter(filter_func)

    print("train len", len(train_dataset))
    print("eval len", len(eval_dataset))

    return train_dataset, eval_dataset


def load_model(model_id):
    quant_config = TorchAoConfig(quant_type="int8_weight_only", group_size=128)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    return model, tokenizer


def main():
    # load quantized model
    model_id = "Qwen/Qwen2.5-Coder-0.5B"
    print("Loading base model...")
    model, tokenizer = load_model(model_id)
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token

    # load dataset
    print("Loading dataset...")
    train_dataset, eval_dataset = load_zeta_dataset(tokenizer, num_samples=64)
    data_collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)

    # lora config
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = prepare_model_for_kbit_training(model)
    print("Model ajsuted for 8bit training...")

    # get model with adapters
    model = get_peft_model(model, peft_config)
    print("Peft model created...")

    print("Trainable Model Parameters: ", model.print_trainable_parameters())

    # put model on compute device
    model.to(DEVICE)
    print("Model put in compute device...")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=WARM_UP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        max_grad_norm=1.0,
        gradient_accumulation_steps=GRAD_ACCM,
        gradient_checkpointing=False,
        save_steps=32,
        eval_steps=32,
        eval_strategy="steps",
        save_strategy="steps",
        dataloader_num_workers=4,
        torch_empty_cache_steps=16,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        optim="adamw_torch_fused",
        # optim="adamw_torch_fused",
        # report_to="wandb",
        # run_name=wandb_run_name,
        logging_steps=10,
        do_eval=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    print("And training starts now...")
    trainer_stats = trainer.train()

    print("Trainer Stats: \n", trainer_stats)

    # save model
    model.save_pretrained(OUTPUT_DIR + "/final/")


if __name__ == "__main__":
    main()
