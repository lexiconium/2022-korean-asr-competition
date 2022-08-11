import argparse
import json
import os
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to the dataset directory")
    args = parser.parse_args()

    answer_path = os.path.join(args.dataset, "answer.json")

    with open(answer_path, "r") as f:
        answer = json.load(f)

    choice_pattern = re.compile("\\([^\\(\\)]+\\)/\\([^\\(\\)]+\\)")
    replace_pattern = re.compile("[^ 가-힣,.!?]+")
    whitespace_pattern = re.compile("\\s+")


    def choose(s_0: str, s_1: str):
        if replace_pattern.findall(s_0):
            return s_1
        return s_0


    for prob_idx, prob in enumerate(answer["Q2"]):
        sentence = prob["original"]

        choices = [
            choose(*map(lambda _s: _s[1:-1], s.split("/")))
            for s in choice_pattern.findall(sentence)
        ]

        pieces = [replace_pattern.sub("", s) for s in choice_pattern.split(sentence)]

        processed = ""
        for piece, choice in zip(pieces, choices):
            processed += piece + choice
        processed += pieces[-1]

        prob["new"] = whitespace_pattern.sub(" ", processed.strip())

    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False, indent=4)
