import os
import subprocess
from typing import List

from pyctcdecode import BeamSearchDecoderCTC, build_ctcdecoder
from transformers import AutoTokenizer

BUILDER_SCRIPT = "build_kenlm_ngrams.sh"


def build_n_gram_decoder(texts: List[str], tokenizer: AutoTokenizer, n_gram: int) -> BeamSearchDecoderCTC:
    textfile = "text.txt"
    with open(textfile, "w") as file:
        file.write(" ".join(texts))

    os.system(f"chmod +x {BUILDER_SCRIPT}")

    output = f"{n_gram}gram.arpa"
    subprocess.run(
        [f"./{BUILDER_SCRIPT}", str(n_gram), textfile, output],
        check=True
    )

    corrected_output = f"corrected_{output}"
    with open(output, "r") as read_file, open(corrected_output, "w") as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count = line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count) + 1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)

    vocab_dict = tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()), kenlm_model_path=corrected_output
    )

    return decoder
