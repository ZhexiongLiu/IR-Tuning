import transformers
import json
import pandas as pd
import argparse
import os
import wandb
import torch
from ist import ISTCallback
from rst import RSTCallback
from ir import IRCallback, CosineCallback, CosineSimilarityTracker, WeightCallback, GradientTracker
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback, BitsAndBytesConfig
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForSequenceClassification
from utilities import generate_prompt_iterater, generate_prompt_erevise_purpose, load_config_file
from utilities import get_iterater_data, get_erevise_purpose_data
from utilities import compute_iterater_metrics, compute_erevise_purpose_metrics
from peft.src.peft import LoraConfig, DoraConfig, BottleneckConfig
from peft.src.peft import get_peft_model, prepare_model_for_int8_training


class TestSetEvaluationCallback(TrainerCallback):
    def __init__(self, trainer, test_dataset, compute_metrics=None, log_steps=0):
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.log_steps = log_steps
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_steps == 0:
            trainer = self.trainer
            test_results = trainer.predict(test_dataset=self.test_dataset)
            for metric, value in test_results.metrics.items():
                wandb.log({f"test_{metric}": value, "global_step": state.global_step})
            print("Test results:\n", test_results.label_ids, "\n", test_results.metrics)


def main():
    if args.dataset_name == 'iterater-human':
        dataset_origin = load_dataset("wanyu/IteraTeR_human_sent")
        compute_metrics = compute_iterater_metrics
        generate_prompt = generate_prompt_iterater

        train_before_sents, train_after_sents, train_labels = get_iterater_data(dataset_origin['train'], (1, 1, 1, 1, 1))
        val_before_sents, val_after_sents, val_labels = get_iterater_data(dataset_origin['validation'], None)
        test_before_sents, test_after_sents, test_labels = get_iterater_data(dataset_origin['test'], None)
        args.num_labels = 5

    elif args.dataset_name == 'erevise-purpose':
        dataset_origin = pd.read_excel("./data/revision_purpose.xlsx").sample(frac=1, random_state=171).reset_index(drop=True)
        compute_metrics = compute_erevise_purpose_metrics
        generate_prompt = generate_prompt_erevise_purpose

        train_df, temp_df = train_test_split(dataset_origin, test_size=0.2, random_state=171)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=171)
        train_before_sents, train_after_sents, train_labels = get_erevise_purpose_data(train_df, (1, 1, 1, 1, 1, 1))
        val_before_sents, val_after_sents, val_labels = get_erevise_purpose_data(val_df, None)
        test_before_sents, test_after_sents, test_labels = get_erevise_purpose_data(test_df, None)
        args.num_labels = 6
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_instructions = [generate_prompt(before, after, context=None) for before, after in zip(train_before_sents, train_after_sents)]
    val_instructions = [generate_prompt_iterater(before, after, context=None) for before, after in zip(val_before_sents, val_after_sents)]
    test_instructions = [generate_prompt_iterater(before, after, context=None) for before, after in zip(test_before_sents, test_after_sents)]

    if args.load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            num_labels=args.num_labels
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            num_labels=args.num_labels
        )

    if args.model_name == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_int8_training(model)

    if args.adapter_name == "lora":
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.adapter_name == "dora":
        config = DoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=True,
            Wdecompose_target_modules=None
        )
    elif args.adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=256,
            non_linearity='tanh',
            adapter_dropout=0.0,
            use_parallel_adapter=True,
            use_adapterp=False,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
            scaling=1.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        raise ValueError(f"Unknown adapter: {args.adapter_name}")

    model = get_peft_model(model, config)
    model.config.pad_token_id = tokenizer.pad_token_id
    print(model)
    model.print_trainable_parameters()

    dataset_train = Dataset.from_dict({"first_sent": train_before_sents, "second_sent": train_after_sents, "label": train_labels, "instruction": train_instructions}).shuffle(seed=171)
    dataset_val = Dataset.from_dict({"first_sent": val_before_sents, "second_sent": val_after_sents, "label": val_labels, "instruction": val_instructions})
    dataset_test = Dataset.from_dict({"first_sent": test_before_sents, "second_sent": test_after_sents, "label": test_labels, "instruction": test_instructions})
    dataset = DatasetDict({'train': dataset_train, 'test': dataset_test, 'validation': dataset_val})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        if args.use_instruction:
            return tokenizer(examples["instruction"], truncation=True, padding="max_length", max_length=args.max_length)
        else:
            return tokenizer(examples["first_sent"], examples["second_sent"], truncation=True, padding="max_length", max_length=args.max_length)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer, return_tensors="pt", padding="max_length", max_length=args.max_length)

    trainer_callbacks = []
    if args.tuning_method == "ist":
        print('IST Tuning activated')
        ist_callback = ISTCallback(model, encoded_dataset["train"].remove_columns(['first_sent', 'second_sent', 'instruction']), data_collator, args.batch_size)
        trainer_callbacks.append(ist_callback)
    elif args.tuning_method == "ir":
        print('IR Tuning activated')
        if args.importance_metric_name == "gradient":
            tracker = GradientTracker(model)
            ir_callback = IRCallback(model, encoded_dataset["train"].remove_columns(['first_sent', 'second_sent', 'instruction']), data_collator, tracker, args.batch_size, args.split_num)
        elif args.importance_metric_name == "cosine":
            tracker = CosineSimilarityTracker(model)
            ir_callback = CosineCallback(model, tracker, args.split_num)
        elif args.importance_metric_name == "weight":
            ir_callback = WeightCallback(model, args.split_num)
        else:
            raise NotImplemented
        trainer_callbacks.append(ir_callback)
    elif args.tuning_method == "rst":
        print('RST activated')
        rst_callback = RSTCallback(model, encoded_dataset["train"].remove_columns(['first_sent', 'second_sent', 'instruction']), data_collator, args.batch_size)
        trainer_callbacks.append(rst_callback)
    elif args.tuning_method == "full":
        print('Full Tuning activated')
        pass
    else:
        raise ValueError(f"Unknown tuning method: {args.tuning_method}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.logging_steps,
        save_strategy="no",
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="wandb",
        max_steps=args.max_steps
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        data_collator=data_collator,
        callbacks=trainer_callbacks,
        compute_metrics=compute_metrics,
    )

    test_callback = TestSetEvaluationCallback(
        trainer=trainer,
        test_dataset=encoded_dataset["test"],
        log_steps=args.logging_steps,
    )
    trainer.add_callback(test_callback)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='llama3.1-8b', choices=['llama3.1-8b', 'llama2-13b', 'mistral-7b', 'deepseek-r1-distil-8b', 'qwen3-8b', 'phi-2'], help='model name')
    parser.add_argument('--dataset-name', default='iterater-human', choices=['iterater-human', 'erevise-purpose'], help='dataset name')
    parser.add_argument('--adapter-name', default='lora', choices=['lora', 'dora', 'bottleneck'], help='model adapter name')
    parser.add_argument('--importance-metric-name', default='gradient', choices=['gradient', 'weight', 'cosine'], help='importance score metric name')
    parser.add_argument('--tuning-method', default='ir', choices=['ir', 'rst', 'ist', 'full'], help='layer-wise tuning method')
    parser.add_argument('--output-dir', default='experiments/debug', help='experiment directory')
    parser.add_argument('--use-instruction', default=False, action='store_true', help='whether to use instruction')
    parser.add_argument('--load-8bit', default=False, action='store_true', help='whether to load 8bit model')

    parser.add_argument('--gpu', type=str, default=0, help='default gpu id')
    parser.add_argument('--seed', type=int, default=171, help='seed number')
    parser.add_argument('--num-labels', type=int, default=5, help='number of labels')
    parser.add_argument('--max-length', type=int, default=256, help='max length of token sequence')

    parser.add_argument('--lora-r', type=int, default=32, help='lora rank')
    parser.add_argument('--lora-alpha', type=int, default=64, help='lora alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='lora dropout')

    parser.add_argument('--warmup-steps', type=int, default=100, help='warmup steps for learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--per-device-train-batch-size', type=int, default=8, help='device batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='gradient accumulation steps')

    parser.add_argument('--num-train-epochs', type=int, default=4, help='number of epochs')
    parser.add_argument('--max-steps', type=int, default=-1, help='number of maximum steps')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.00, help='weight decay')
    parser.add_argument('--logging-steps', type=int, default=10, help='log result every n steps')
    parser.add_argument('--save-strategy', type=str, default='no', help='save strategy')
    parser.add_argument('--save-steps', type=int, default=0, help='save model every n steps')
    parser.add_argument('--split-num', type=int, default=1, help='number of splits')

    args = parser.parse_args()
    config = load_config_file('config.yaml')
    huggingface_key = config['huggingface_key']
    wandb_key = config['wandb_key']
    wandb.login(key=wandb_key)

    prefix = "ir-tuning"
    if args.tuning_method == 'ir':
        wandb_name = f"{prefix}" + '-' + args.model_name + '-' + args.adapter_name + '-' + args.tuning_method + '-' + args.importance_metric_name + f'-split{args.split_num}'
    else:
        wandb_name = f"{prefix}" + '-' + args.model_name + '-' + args.adapter_name + '-' + args.tuning_method
    wandb_project = f"{prefix}" + '-' + args.dataset_name
    args.output_dir = os.path.join('experiments', wandb_project, wandb_name)

    os.environ["WANDB_NAME"] = wandb_name
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["HUGGING_FACE_HUB_TOKEN"] = huggingface_key
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.use_instruction:
        os.environ["WANDB_NAME"] += '-instruction'
        args.output_dir += '-instruction'
        args.max_length = 256
        args.per_device_train_batch_size = 16

    args.gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_name == 'llama3.1-8b' and args.use_instruction:
        args.base_model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model_name == 'llama3.1-8b' and not args.use_instruction:
        args.base_model = 'meta-llama/Llama-3.1-8B'
    elif args.model_name == 'llama2-13b':
        args.base_model = 'meta-llama/Llama-2-13b-hf'
        args.batch_size = 8
        args.per_device_train_batch_size = 8
    elif args.model_name == 'mistral-7b' and args.use_instruction:
        args.base_model = 'mistralai/Mistral-7B-Instruct-v0.3'
    elif args.model_name == 'mistral-7b' and not args.use_instruction:
        args.base_model = 'mistralai/Mistral-7B-v0.3'
    elif args.model_name == 'deepseek-r1-distil-8b':
        args.base_model = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    elif args.model_name == 'qwen3-8b' and args.use_instruction:
        args.base_model = 'Qwen/Qwen3-8B'
    elif args.model_name == 'qwen3-8b' and not args.use_instruction:
        args.base_model = 'Qwen/Qwen3-8B'
    elif args.model_name == 'phi-2':
        args.base_model = 'microsoft/phi-2'
    else:
        raise ValueError('Unknown model name')

    with open(os.path.join(args.output_dir, "parameter.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # sys.stdout = open(os.path.join(args.output_dir, "output.log"), "w")
    # sys.stderr = open(os.path.join(args.output_dir, "output.log"), "w")

    main()
    print("Task completed successfully!")
