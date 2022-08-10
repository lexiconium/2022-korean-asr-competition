import argparse
import json
import os
import struct


def extract_info(filename: str, verbose: bool = False):
    info = {}
    with open(filename, "rb") as f:
        riff, overall_size, file_type = struct.unpack("<4sI4s", f.read(12))
        format_chunk_marker, format_chunk_length = struct.unpack("<4sI", f.read(8))
        format_type, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
            "<HHIIHH", f.read(16)
        )

        info["byte_rate"] = byte_rate

        if verbose:
            print(
                f"riff: {riff}, overall_size: {overall_size}, file_type: {file_type}\n"
                f"format_chunk_marker: {format_chunk_marker}, format_chunk_length: {format_chunk_length}\n"
                f"format_type: {format_type}, channels: {channels}, sample_rate: {sample_rate}\n"
                f"byte_rate: {byte_rate}, block_align: {block_align}, bits_per_sample: {bits_per_sample}"
            )

        offset = f.tell()
        while offset < overall_size:
            sub_chunk_marker, sub_chunk_length = struct.unpack("<4sI", f.read(8))
            sub_chunk_marker = sub_chunk_marker.decode("utf-8")
            info[sub_chunk_marker] = f.read(sub_chunk_length)
            info[f"{sub_chunk_marker}_length"] = sub_chunk_length

            if verbose:
                print(f"sub_chunk_marker: {sub_chunk_marker}, sub_chunk_length: {sub_chunk_length}")

            offset += sub_chunk_length + 8

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to the dataset directory")
    parser.add_argument("--verbose", type=bool, default=False, help="whether to print extra information")
    args = parser.parse_args()

    prob1_dir = os.path.join(args.dataset, "문제1")
    answer_path = os.path.join(args.dataset, "answer.json")

    with open(answer_path, "r") as f:
        answer = json.load(f)

    for prob_idx, prob in enumerate(answer["Q1"]):
        filepath = os.path.join(prob1_dir, f"{prob['filename']}")

        info = extract_info(filepath, verbose=args.verbose)
        prob["duration"] = float(f"{info['data_length'] / info['byte_rate']:.3f}")

        if "THIS" in info:
            try:
                prob["THIS"] = info['THIS'].decode('utf-8')
            except:
                prob["THIS"] = float(f"{info['THIS_length'] / info['byte_rate']:.3f}")

    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False, indent=4)
