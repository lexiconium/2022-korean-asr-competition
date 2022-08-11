1. 스크립트
solve.py 스크립트는 다음 구조를 가정한다.

dataSet
├── answer.json
├── 문제1
└── 문제3

스크립트는 문장으로부터 ()/() 형태의 선택지를 읽어 한글과 문장부호만을 포함한 선택지를 선택한다.
이후 선택지가 제거된 원래의 문장에서 한글과 문장부호를 제외한 요소들을 제거하고 위의 선택지를 빈칸에 넣는다.
마지막으로 불필요한 문장을 제거한 뒤 answer.json에 기입한다.

다음과 같이 실행한다.
python solve.py --dataset <path-to-dataset>


2. Environment
macOS Monterey 12.5
python 3.10.4
