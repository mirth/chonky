import numpy as np
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

torch.set_float32_matmul_precision("high")


def distilbert(model_name):
    id2label = {
        0: "O",
        1: "separator",
    }
    label2id = {
        "O": 0,
        "separator": 1,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def modernbert(model_id):
    id2label = {
        0: "O",
        1: "separator",
    }
    label2id = {
        "O": 0,
        "separator": 1,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        _attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def main(dataset_id, model_id, output_dir, batch_size, max_seq_len=None):
    model, tokenizer = modernbert(model_id=model_id)
    # model, tokenizer = distilbert(model_name=model_id)

    if max_seq_len is None:
        max_seq_len = tokenizer.model_max_length

    dataset = load_from_disk(dataset_id)
    print(dataset)
    dataset_val = dataset["test"]
    dataset_train = dataset["train"]
    label_list = ["O", "separator"]
    seqeval = evaluate.load("seqeval")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_seq_len,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    tokenized_dataset_train = dataset_train.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset_val = dataset_val.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=25,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main(
        dataset_id="data/refined-bookcorpus-dataset_hf_ModernBERT-base1024",
        model_id="modernbert_large_minipile/checkpoint-54717",
        output_dir="modernbert_large_minipile_plus_bookcorpus",
        batch_size=64,
        max_seq_len=1024,
    )
