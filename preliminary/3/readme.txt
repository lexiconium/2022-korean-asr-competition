1. 함수
torchaudio.functional.filtering의 vad 함수에서 필터링하는 지점의 인덱스를 추가적으로 반환하도록 바꾼 함수이다.


2. 스크립트
solve.py 스크립트는 다음 구조를 가정한다.

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

스크립트는 answer.json의 Q3로부터 문제에 사용되는 파일명을 읽은 후 voice_activity_detection 함수를 사용하여 voice activity의 시작지점과
끝지점을 구한다.

다음과 같이 실행한다.
python solve.py --dataset <absolute-path-to-dataSet>


3. Dependencies
torch 1.12.0
torchaudio 0.12.0


4. Environment
macOS Monterey 12.5
python 3.10.4


4-1.
torchaudio는 운영체제에 따라 다음 backend를 사용한다.
“sox_io” (default on Linux/macOS)
“soundfile” (default on Windows)
