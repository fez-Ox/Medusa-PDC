# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

import json
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from safetensors.torch import save_file
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
)
from transformers import (
    TrainingArguments as HFTrainingArguments,
)
from transformers.trainer_pt_utils import LabelSmoother

from medusa.model.medusa_model_legacy import MedusaConfig, MedusaModel

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def _is_rank0():
    return os.environ.get("RANK", "0") == "0"


def _cuda_is_ampere_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def _cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        return torch.cuda.is_bf16_supported()
    return False


def _select_model_dtype(training_args):
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return torch.float32


# Customized for training Medusa heads
class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            medusa = model.module.medusa
        else:
            medusa = model.medusa

        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(2 + i)].contiguous()
            medusa_labels = labels[..., 2 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 2):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
        self.log(log)
        return (loss, logits) if return_outputs else loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )

    def __post_init__(self):
        if getattr(self, "tf32", False) and not _cuda_is_ampere_or_newer():
            if _is_rank0():
                print(
                    "Warning: --tf32=True is not supported on this system; falling back to --tf32=False."
                )
            self.tf32 = False

        if getattr(self, "bf16", False) and not _cuda_supports_bf16():
            if _is_rank0():
                print(
                    "Warning: --bf16=True is not supported on this system; falling back to --bf16=False."
                )
            self.bf16 = False
            if torch.cuda.is_available() and not getattr(self, "fp16", False):
                if _is_rank0():
                    print(
                        "Warning: Enabling --fp16=True for T4-compatible mixed precision."
                    )
                self.fp16 = True

        try:
            super().__post_init__()
        except ValueError as exc:
            error_text = str(exc)
            if (
                getattr(self, "tf32", False)
                and "--tf32 requires Ampere or a newer GPU arch" in error_text
            ):
                if _is_rank0():
                    print(
                        "Warning: --tf32=True is not supported on this system; falling back to --tf32=False."
                    )
                self.tf32 = False
                super().__post_init__()
                return
            if getattr(self, "bf16", False) and "bf16" in error_text.lower():
                if _is_rank0():
                    print(
                        "Warning: --bf16=True is not supported on this system; falling back to --bf16=False."
                    )
                self.bf16 = False
                if torch.cuda.is_available() and not getattr(self, "fp16", False):
                    if _is_rank0():
                        print(
                            "Warning: Enabling --fp16=True for T4-compatible mixed precision."
                        )
                    self.fp16 = True
                super().__post_init__()
                return
            raise


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []
    # # import pdb; pdb.set_trace()
    for i, conversation in enumerate(sources):
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompts.append(prompt)
        conversations.append(conversation)

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(
        zip(conversations, targets, prompts)
    ):
        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"]
                # Unfortunate strip() necessary because chat templates are doing the same.
                start = prompt.index(content.strip())
                stop = start + len(content)
                indices = []
                for tok_index, (tok_start, tok_stop) in enumerate(
                    encoding.offset_mapping[conv_index]
                ):
                    if tok_stop >= start or tok_start < tok_stop:
                        indices.append(tok_index)
                target[indices] = encoding.input_ids[conv_index][indices]

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if model_args.load_in_8bit and model_args.load_in_4bit:
        raise ValueError(
            "Only one of --load_in_8bit and --load_in_4bit can be enabled."
        )

    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # Add this block right after tokenizer loading
    if tokenizer.chat_template is None:
        # Vicuna-style chat template
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "USER: {{ message['content'] }} "
            "{% elif message['role'] == 'assistant' %}"
            "ASSISTANT: {{ message['content'] }}</s>"
            "{% endif %}"
            "{% endfor %}"
        )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token

    # Making sure the tokenizer works before loading the model.
    #    print(tokenizer(["This is a test", "secondary"], padding=True))
    #    print(tokenizer.apply_chat_template([{"role": "user", "content": "This is a test"}]))

    # Load model and tokenizer
    model_dtype = _select_model_dtype(training_args)
    if _is_rank0():
        print(f"Loading base model with dtype={model_dtype}.")

    quantization_kwargs = {}
    if model_args.load_in_8bit:
        quantization_kwargs["load_in_8bit"] = True
    if model_args.load_in_4bit:
        quantization_kwargs["load_in_4bit"] = True
    if quantization_kwargs:
        quantization_kwargs["device_map"] = "auto"
        if _is_rank0():
            print(f"Enabling quantized loading with arguments: {quantization_kwargs}.")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=model_dtype,
        **quantization_kwargs,
    )

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Add Medusa heads
    medusa_lm_head = MedusaModel(
        model,
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}_medusa_mlp_{model_args.model_name_or_path.split('/')[-1]}_medusa_{training_args.medusa_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.medusa_num_layers}"

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Generate Medusa config for pushing to HF hub
    medusa_config = MedusaConfig(
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
        version="2",
    )

    # Save Medusa config
    medusa_config.save_pretrained(training_args.output_dir)

    # Start trainner
    trainer = CustomizedTrainer(
        model=medusa_lm_head, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # Save MedusaHead seperately
    if hasattr(medusa_lm_head, "module"):
        lm_head = medusa_lm_head.module.medusa_head
    else:
        lm_head = medusa_lm_head.medusa_head
    import deepspeed

    with deepspeed.zero.GatheredParameters(lm_head.parameters()):
        state_dict = lm_head.state_dict()

    # Save Medusa heads
    if local_rank == 0:
        # Modify the tokenizer internal state before saving.
        tokenizer.encode("Test", truncation=None, padding="do_not_pad")
        tokenizer.save_pretrained(training_args.output_dir)
        save_file(
            state_dict,
            os.path.join(training_args.output_dir, "medusa_lm_head.safetensors"),
        )


if __name__ == "__main__":
    train()
