from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def split_into_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def split_text_into_even_chunks(tokenizer, text):
    ids = tokenizer(text, truncation=False, add_special_tokens=False)
    ids = ids["input_ids"]
    ids = split_into_chunks(ids, n=tokenizer.model_max_length)
    chunks = (tokenizer.decode(chunk_ids) for chunk_ids in ids)

    return chunks


def split_into_semantic_chunks(text, ners):
    begin_index = 0

    for i, _c in enumerate(text):
        for ner in ners:
            if i == ner["end"]:
                chunk = text[begin_index : ner["end"]]
                chunk = chunk.strip()
                yield chunk

                begin_index = ner["end"]

    yield text[begin_index:]


class Chonky:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        id2label = {
            0: "O",
            1: "separator",
        }
        label2id = {
            "O": 0,
            "separator": 1,
        }

        model = AutoModelForTokenClassification.from_pretrained(
            "./bookcorpus_model_5m/checkpoint-26520",
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        model.to(device)
        self.pipe = pipeline(
            "ner",
            model=model,
            tokenizer=self.tokenizer,
            device=device,
            aggregation_strategy="simple",
        )

    def __call__(self, text):
        text_chunks = split_text_into_even_chunks(self.tokenizer, text)

        for text_chunk in text_chunks:
            output = self.pipe(text_chunk)

            yield from split_into_semantic_chunks(text_chunk, output)


if __name__ == "__main__":
    with open(
        "data/paul_graham_essay_no_new_line/paul_graham_essay_no_new_line.txt"
    ) as file:
        pg = file.read()

    c = Chonky(device="cuda")
    for sem_chunk in c(pg):
        print(sem_chunk)
        print("--")
