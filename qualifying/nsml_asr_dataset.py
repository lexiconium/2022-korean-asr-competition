import os

import datasets
import nsml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_DESCRIPTION = "Automatic speech recognition dataset for NSML(NAVER Smart Machine Learning)"

_DATA_DIR = os.path.join(nsml.DATASET_PATH, "train", "train_data")
_METADATA_PATH = os.path.join(nsml.DATASET_PATH, "train", "train_label")


class PCMAudio(datasets.Audio):
    def _decode_non_mp3_path_like(self, path, **kwargs):
        try:
            import librosa
        except ImportError as err:
            raise ImportError("To support decoding audio files, please install 'librosa'.") from err

        with open(path, "rb") as f:
            buf = f.read()
            pcm_data = np.frombuffer(buf, dtype="int16")
            array = librosa.util.buf_to_float(pcm_data, n_bytes=2)

        return array, self.sampling_rate


class NSMLAutomaticSpeechRecognitionDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "audio": PCMAudio(sampling_rate=16_000, decode=False),
                    "transcript": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        metadata = pd.read_csv(_METADATA_PATH)
        train_metadata, validation_metadata = train_test_split(
            metadata, test_size=0.05, random_state=42, shuffle=True
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"metadata": train_metadata}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"metadata": validation_metadata}
            )
        ]

    def _generate_examples(self, metadata: pd.DataFrame):
        for row in metadata.itertuples(index=False):
            yield row.filename, {
                "audio": os.path.join(_DATA_DIR, f"{row.filename}"),
                "transcript": row.text
            }
