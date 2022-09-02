# Copyright 2022 Minsoo Kim <min-soo.kim@outlook.kr>
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#
# This file has been modified by Minsoo Kim <min-soo.kim@outlook.kr>.

""" Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

import functools
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Union

import evaluate
import nsml
import numpy as np
import torch
import transformers
import wandb
from datasets import DatasetDict, load_dataset
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer, HfArgumentParser, Trainer,
                          TrainerCallback, TrainerControl, TrainerState, TrainingArguments, Wav2Vec2ForCTC,
                          Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, get_scheduler, set_seed)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from decoder.n_gram_decoder import build_n_gram_decoder
from nsml_asr_dataset import PCMAudio
from processors.modeling_text_processors import (ChoiceSelectionTextProcessor,
                                                 DuplicatedWhitespaceRemovingTextProcessor,
                                                 SequentialTextProcessor)
from utils.config import read_yaml_config

# Will error if the minimal version of Transformers is not installed
check_min_version("4.21.0")

require_version("datasets>=1.18.0")

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector"
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )
    use_lm: bool = field(default=False, metadata={"help": "Whether or not to use a language model decoder."})
    n_gram: int = field(
        default=4,
        metadata={"help": "Number of grams for n-gram language model decoder. Valid only if use_lm is true."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train+validation'"
            )
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="transcript",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    chars_to_ignore: Optional[List[str]] = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: List[str] = list_field(
        default=["wer", "cer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "If :obj:`True`, will use the token generated when running"
                ":obj:`huggingface-cli login` as HTTP bearer authorization for remote files."
            )
        },
    )
    unk_token: str = field(
        default="<unk>",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="<pad>",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The target language that should be used be"
                " passed to the tokenizer for tokenization. Note that"
                " this is only relevant if the model classifies the"
                " input audio to a sequence of phoneme sequences."
            )
        },
    )

    def __post_init__(self):
        if self.preprocessing_num_workers is None:
            self.preprocessing_num_workers = len(os.sched_getaffinity(0))


@dataclass
class CTCTrainingArguments(TrainingArguments):
    wandb_authentication_path: str = field(
        default="authentication/wandb.yaml", metadata={"help": "Path to WandB authentication file."}
    )
    retain_pretraining_configs: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to retain pretraining configuration e.g. dropouts."
        }
    )
    use_tri_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use tri-state learning rate scheduler."
                " Applied over lr_scheduler_type if true."
            )
        }
    )


@dataclass
class NSMLArguments:
    mode: str = field(default="train")
    iteration: str = field(default="0")
    pause: int = field(default=0)


class NSMLCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = math.ceil(state.epoch)

        def save(ckpt_dir: str, **nsml_kwargs):
            kwargs["model"].save_pretrained(ckpt_dir)
            nsml_kwargs["processor"].save_pretrained(ckpt_dir)
            torch.save(kwargs["optimizer"].state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            torch.save(kwargs["lr_scheduler"].state_dict(), os.path.join(ckpt_dir, "lr_scheduler.pt"))

            logger.info(f"epoch {epoch} saved in {ckpt_dir}")

        nsml.save(epoch, save_fn=save)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        nsml.report(
            summary=True,
            **kwargs["metrics"]
        )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Union[Wav2Vec2Processor, Wav2Vec2ProcessorWithLM]
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # create vocabulary given training and test labels
    def extract_all_chars(batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names
    )

    # take union of all unique characters in each dataset
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]),
        vocabs.values()
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


def bind_model(state: dict, processor):
    def save(path, *args, **kwargs):
        """
        NSML save function is defined and called inside NSMLCallback
        """

    def load(path, *args, **kwargs):
        logger.info(
            f"Only model and processor are called from the checkpoint {path}."
            " Optimizer and learning rate scheduler are newly initialized for fork process"
        )

        state["model"] = Wav2Vec2ForCTC.from_pretrained(path)
        try:
            state["processor"] = Wav2Vec2ProcessorWithLM.from_pretrained(path)
        except UserWarning("Wav2Vec2ProcessorWithLM object not found. Loading object from Wav2Vec2Processor."):
            state["processor"] = Wav2Vec2Processor.from_pretrained(path)

    def infer(path, *args, **kwargs):
        decoder = PCMAudio(sampling_rate=16_000)._decode_non_mp3_path_like
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = state["model"].to(device)
        processor = state["processor"]

        results = []

        model.eval()
        with torch.no_grad():
            for filepath in tqdm(glob(os.path.join(path, "*"))):
                array, _ = decoder(filepath)
                array = torch.tensor(array, device=device).unsqueeze(dim=0)
                logits = model(array).logits

                pred_ids = torch.argmax(logits, dim=-1)
                pred_text = processor.batch_decode(pred_ids)[0]

                results.append(
                    {
                        "filename": filepath.split("/")[-1],
                        "text": pred_text
                    }
                )

        return sorted(results, key=lambda x: x["filename"])

    nsml.bind(save=save, load=load, infer=infer, processor=processor)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CTCTrainingArguments, NSMLArguments))
    model_args, data_args, training_args, nsml_args = parser.parse_args_into_dataclasses()

    if nsml_args.mode == "train":
        # Configure Weight and Biases logger
        wandb_authentication = read_yaml_config(training_args.wandb_authentication_path)
        os.system(f"wandb login {wandb_authentication['hash']}")
        wandb.init(
            project="ko_asr",
            entity=wandb_authentication["username"],
            name=model_args.model_name_or_path,
            group="ctc"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity of Transformers logger to info (on main process only)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model
    set_seed(training_args.seed)

    raw_datasets = DatasetDict()
    if nsml_args.mode == "train":
        # Load dataset
        if training_args.do_train:
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.train_split_name,
                use_auth_token=data_args.use_auth_token
            )

            if data_args.audio_column_name not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                    " Make sure to set `--audio_column_name` to the correct audio column - one of"
                    f" {', '.join(raw_datasets['train'].column_names)}."
                )

            if data_args.text_column_name not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                    "Make sure to set `--text_column_name` to the correct text column - one of "
                    f"{', '.join(raw_datasets['train'].column_names)}."
                )

            if data_args.max_train_samples is not None:
                raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

        if training_args.do_eval:
            raw_datasets["eval"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.eval_split_name,
                use_auth_token=data_args.use_auth_token
            )

            if data_args.max_eval_samples is not None:
                raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

        # Preprocess transcripts
        chars_to_ignore_regex = (
            f'[{"".join(data_args.chars_to_ignore)}]' if data_args.chars_to_ignore is not None else None
        )

        text_preprocessor = SequentialTextProcessor(
            ChoiceSelectionTextProcessor(condition="[^ 가-힣]+"),
            DuplicatedWhitespaceRemovingTextProcessor()
        )

        text_column_name = data_args.text_column_name

        def preprocess_transcript(batch):
            if chars_to_ignore_regex is not None:
                batch[text_column_name] = re.sub(chars_to_ignore_regex, "", batch[text_column_name])

            batch["text"] = batch[text_column_name].lower()
            batch["text"] = text_preprocessor(batch["text"])

            return batch

        with training_args.main_process_first(desc="dataset map special characters removal"):
            raw_datasets = raw_datasets.map(
                preprocess_transcript,
                remove_columns=[text_column_name],
                desc="remove special characters from datasets",
            )

    config = None
    tokenizer = None
    decoder = None
    if nsml_args.mode == "train" and not nsml_args.pause:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=data_args.use_auth_token
        )

        # create vocabulary and initialize tokenizer
        word_delimiter_token = data_args.word_delimiter_token
        unk_token = data_args.unk_token
        pad_token = data_args.pad_token

        tokenizer_name_or_path = model_args.tokenizer_name_or_path
        tokenizer_kwargs = {}
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = training_args.output_dir
            vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

            with training_args.main_process_first():
                if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
                    try:
                        os.remove(vocab_file)
                    except OSError:
                        # in shared file-systems it might be the case that
                        # two processes try to delete the vocab file at the same time
                        pass

            with training_args.main_process_first(desc="dataset map vocabulary creation"):
                if not os.path.isfile(vocab_file):
                    logger.info("vocabulary file not found - creating it")
                    os.makedirs(tokenizer_name_or_path, exist_ok=True)
                    vocab_dict = create_vocabulary_from_data(
                        raw_datasets,
                        word_delimiter_token=word_delimiter_token,
                        unk_token=unk_token,
                        pad_token=pad_token,
                    )

                    # save vocab dict to be loaded into tokenizer
                    with open(vocab_file, "w") as file:
                        json.dump(vocab_dict, file, ensure_ascii=False)

            tokenizer_kwargs = {
                "config": config if config.tokenizer_class is not None else None,
                "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
                "unk_token": unk_token,
                "pad_token": pad_token,
                "word_delimiter_token": word_delimiter_token,
            }

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_auth_token=data_args.use_auth_token,
            **tokenizer_kwargs
        )

        # update config
        config.update({
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer)
        })
        if not training_args.retain_pretraining_configs:
            config.update({
                "feat_proj_dropout": model_args.feat_proj_dropout,
                "attention_dropout": model_args.attention_dropout,
                "hidden_dropout": model_args.hidden_dropout,
                "final_dropout": model_args.final_dropout,
                "mask_time_prob": model_args.mask_time_prob,
                "mask_time_length": model_args.mask_time_length,
                "mask_feature_prob": model_args.mask_feature_prob,
                "mask_feature_length": model_args.mask_feature_length,
                "layerdrop": model_args.layerdrop,
                "ctc_loss_reduction": model_args.ctc_loss_reduction,
                "activation_dropout": model_args.activation_dropout
            })

        if model_args.use_lm:
            with training_args.main_process_first(desc="building n-gram decoder"):
                # build n-gram language model
                decoder = build_n_gram_decoder(
                    raw_datasets["train"]["text"], tokenizer=tokenizer, n_gram=model_args.n_gram
                )

    feature_extractor = None
    if nsml_args.mode == "train":
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=data_args.use_auth_token
        )

        # prepare dataset
        dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, PCMAudio(sampling_rate=dataset_sampling_rate)
        )
        # make sure that dataset decodes audio with correct sampling rate
        if dataset_sampling_rate != feature_extractor.sampling_rate:
            raw_datasets = raw_datasets.cast_column(
                data_args.audio_column_name, PCMAudio(sampling_rate=feature_extractor.sampling_rate)
            )

        # derive max & min input length for sample rate & max duration
        max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
        min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
        audio_column_name = data_args.audio_column_name
        num_workers = data_args.preprocessing_num_workers

        # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
        phoneme_language = data_args.phoneme_language

        def prepare_dataset(batch):
            sample = batch[audio_column_name]

            inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(batch["input_values"])

            additional_kwargs = {}
            if phoneme_language is not None:
                additional_kwargs["phonemizer_lang"] = phoneme_language

            batch["labels"] = tokenizer(batch["text"], **additional_kwargs).input_ids
            return batch

        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=next(iter(raw_datasets.values())).column_names,
                num_proc=num_workers,
                desc="preprocess datasets"
            )

            def is_audio_in_length_range(length):
                return min_input_length < length < max_input_length

            # filter data that is shorter than min_input_length
            vectorized_datasets = vectorized_datasets.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"]
            )

    processor = None
    model = None
    if nsml_args.mode == "train" and not nsml_args.pause:
        if model_args.use_lm:
            processor = Wav2Vec2ProcessorWithLM(feature_extractor, tokenizer, decoder)
        else:
            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

        model = AutoModelForCTC.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            use_auth_token=data_args.use_auth_token
        )

    state = {}
    bind_model(state, processor=processor)

    if nsml_args.pause:
        nsml.paused(scope=locals())

        if nsml_args.mode == "train":
            model = state["model"]
            processor = state["processor"]

    if nsml_args.mode == "train":
        if model_args.freeze_feature_encoder:
            model.freeze_feature_encoder()

        # initialize data collator
        data_collator = DataCollatorCTCWithPadding(processor=processor)

        # set metrics
        eval_metrics = {metric: evaluate.load(metric) for metric in data_args.eval_metrics}

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

            pred_str = tokenizer.batch_decode(pred_ids)
            # we do not want to group tokens when computing the metrics
            label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

            metrics = {
                k: v.compute(predictions=pred_str, references=label_str)
                for k, v in eval_metrics.items()
            }

            return metrics

        # initialize optimizer and learning rate scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay
        )

        max_steps = training_args.max_steps
        warmup_steps = training_args.warmup_steps

        if not max_steps > 0:
            dummy_loader = DataLoader(
                vectorized_datasets["train"],
                batch_size=training_args.per_device_train_batch_size
            )
            max_steps = len(dummy_loader) * training_args.num_train_epochs
            max_steps /= (training_args.gradient_accumulation_steps * training_args.n_gpu)

            del dummy_loader

        if warmup_steps == 0 and training_args.warmup_ratio > 0:
            warmup_steps = int(max_steps * training_args.warmup_ratio)

        def get_tri_state_schedule(optimizer, num_training_steps, last_epoch=-1):
            """
            Create a schedule with a learning rate that is warmed up for the first 10% of updates,
            held constant for the next 40% and then linearly decayed for the remainder. [wav2vec 2.0: A Framework for
            Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477.pdf)

            Args:
                optimizer ([`~torch.optim.Optimizer`]):
                    The optimizer for which to schedule the learning rate.
                num_training_steps (`int`):
                    The total number of training steps.
                last_epoch (`int`, *optional*, defaults to -1):
                    The index of the last epoch when resuming training.

            Return:
                `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
            """

            logger.info(
                f"tri-state learning rate scheduler ignores "
                f"user input warmup steps {training_args.warmup_steps} and ratio {training_args.warmup_steps}"
            )

            num_warmup_steps = int(num_training_steps * 0.1)
            num_plateau_steps = int(num_training_steps * 0.4)
            num_decreasing_steps = num_training_steps - num_warmup_steps - num_plateau_steps

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                if current_step < num_warmup_steps + num_plateau_steps:
                    return 1
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_decreasing_steps))
                )

            return LambdaLR(optimizer, lr_lambda, last_epoch)

        if training_args.use_tri_lr_scheduler:
            scheduler = get_tri_state_schedule(optimizer, max_steps)
        else:
            scheduler = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer,
                warmup_steps,
                max_steps
            )

        # initialize trainer
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
            eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
            tokenizer=feature_extractor,
            optimizers=(optimizer, scheduler),
            callbacks=[NSMLCallback]
        )

        if training_args.do_train:
            train_result = trainer.train()

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(vectorized_datasets["train"])
            )
            metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))
            trainer.log_metrics("train", metrics)

            torch.cuda.empty_cache()

        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None else
                len(vectorized_datasets["eval"])
            )
            metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))
            trainer.log_metrics("eval", metrics)


if __name__ == "__main__":
    main()
