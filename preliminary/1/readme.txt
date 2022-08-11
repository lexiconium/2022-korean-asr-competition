1. 함수
solve.py 안의 extract_info 함수는 형식(https://docs.fileformat.com/audio/wav/)에 따라 wav 파일의 정보를 얽는다.
이때 byte rate(Sample Rate * BitsPerSample * Channels) / 8)을 sub chunk의 내용물, 길이와 함께 반환한다.


2. 스크립트
solve.py 스크립트는 다음 구조를 가정한다.

dataSet
├── answer.json
├── 문제1
│ ├── 문제1-1.wav
│ ├── 문제1-2.wav
│ ├── .
│ ├── .
│ ├── .
│ └── 문제1-n.wav
└── 문제3

스크립트는 answer.json의 Q1으로부터 문제에 사용되는 파일명을 읽은 후 extract_info 함수를 사용하여 duration(size / byte rate)을 구한다.
THIS sub chunk가 존재할 경우 우선 utf-8 decoding을 시도한 뒤, 불가능할 경우 chunk의 duration을 계산한다.
마지막으로 이를 answer.json에 기입한다.

다음과 같이 실행한다.
python solve.py --dataset <path-to-dataset>


3. Environment
macOS Monterey 12.5
python 3.10.4
