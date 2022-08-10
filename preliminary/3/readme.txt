1. 함수
torchaudio.functional.filtering의 vad 함수에서 필터링하는 지점의 인덱스를 추가적으로 반환하도록 바꾼 함수이다.



2. 스크립트 실행
solve.py 스크립트를 실행할 땐 다음 구조를 가정한다.

dataSet
├── answer.json
├── 문제1
└── 문제3
    ├── 문제3-1.wav
    ├── 문제3-2.wav
    ├── .
    ├── .
    ├── .
    └── 문제3-n.wav

다음과 같이 실행한다.
python solve.py --dataset <path-to-dataset>


3. Dependencies
python 3.10.4
torch 1.12.0
torchaudio 0.12.0


4. Environment
torchaudio는 운영체제에 따라 다음 backend를 사용한다.
“sox_io” (default on Linux/macOS)
“soundfile” (default on Windows)

Q3에 대한 answer는 macOS Monterey 12.5에서 얻었다.
