# Graph Context Manager(GCM)
제조/물류 환경에서 다중 로봇 작업을 위한 시-공간 작업 맥락 그래프의 생성과 맥락 추론기 개발

![GCM Model Structure](assets/GraphContextManagerModel.png)

## Description
협업 상대 로봇의 행동 맥락(Action Context) 정보를 추론하는 그래프 신경망 기반 맥락 추론 모델입니다.


## Features
- **데이터 수집 및 처리**: Isaac 시뮬레이터에서 직접 수집한 데이터셋 이용
- **모델 구현**: GCN, TripletGCN 그래프 신경망 모델 이용
- **결과 시각화**: Graphviz API 활용하여 학습된 모델의 추론 결과를 시각화 및 저장 

---

## 📂 Directory

├── datasets/                        # 데이터 로더 및 원본 데이터
│   ├── GCMDataLoader.py             # 데이터 로더 메인 코드
│   ├── raw/                         # 원본 데이터 디렉토리
│   │   ├── Isaac/                   # Isaac 데이터셋
│   │   └── MOS/                     # MOS 데이터셋
├── models/                          # 네트워크 모델 정의
│   ├── CloudGCM_Network.py          # 클라우드 기반 GCM 네트워크
│   ├── network_RelNet.py            # RelNet 신경망
│   ├── TripleNetGCN.py              # TripleNet GCN 모델
│   ├── TT_GCN.py                    # TT GCN 모델
│   └── utils/                       # 유틸리티 함수 및 스크립트
│       ├── Graph_Vis.py             # 그래프 시각화 코드
│       ├── visualization.py         # 시각화 유틸리티
│       └── op_utils.py              # 기타 유틸리티 함수
├── rule_based_contextManager/       # 규칙 기반 맥락 추론 모듈
│   └── RuleContextManager.py        # 규칙 기반 추론 로직
├── data_collecter/                  # 데이터 수집 모듈
│   └── DataCollecter.py             # 데이터 수집 및 전처리 코드
├── GCM_main.py                      # 메인 실행 스크립트

---

## 🛠️ Dependencies
    
    ```bash
    conda create -n gcmAgent python=3.8
    conda activate gcmAgent
    pip install -r requirements.txt
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
    pip install torch-geometric
    '''
    
## Run Code
    ```bash
    conda create -n gcmAgent python=3.8

