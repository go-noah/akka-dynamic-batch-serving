# Akka Dynamic Batch Tensorflow Cosine Similarity


## Spec
- **Java 11 (Java 1.8 Support)**
- **Scala 2.12.15**
- **SBT 1.6.2**
- **logback-classic 1.2.10**
- **akka-actor-typed 2.6.8**
- **akka-stream 2.6.8**
- **akka-http 10.2.7**
- **akka-http-caching 10.2.7**
- **akka-http-spray-json 10.2.7**
- **tensorflow-core-platform-gpu 0.4.1**

banchmark
----------------------------------------------------------------
```
CPU : Ryzen 9 3950X
GPU : Nvidia Geforce rtx 3090 24GB (495.29.05)
ENV : docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04
```


```
number of target vectors = 100,000
table value = RPS(request per second)
table value 0 -> Cuda Out Of Memory

| batch/dim |  100 |  300 |  512 | 768 | 1024 | 2048 |
|-----------|-----:|-----:|-----:|----:|-----:|-----:|
|         1 |  393 |  389 |  383 | 365 |  355 |  192 |
|         2 |  822 |  705 |  555 | 432 |  394 |  241 |
|         4 | 1420 |  968 |  684 | 505 |  459 |  260 |
|         8 | 2187 | 1242 |  836 | 605 |  506 |    0 |
|        16 | 3006 | 1442 |  936 | 655 |    0 |    0 |
|        32 | 3750 | 1579 | 1001 |   0 |    0 |    0 |
|        64 | 4184 | 1645 |    0 |   0 |    0 |    0 |
|       128 | 4424 |    0 |    0 |   0 |    0 |    0 |

```


## Introduction

본 프로젝트는 tensorflow-java 사용하여 jvm 환경에서 GPU로 연산을 가속하고 서비스 api로 활용하는 내용을 중심으로 구성되어 있습니다.
모든 구현은 가장 Naive 방식을 선택하였고 GPU와 Dynamic Batch의 활용을 설명하고 실제로 간단히 구현 하는데 초점을 맞추고 있습니다.

Cosine Similarity는 일반적으로 공간에 주어진 단위 벡터의 내적을 통하여 계산이 되고 -1 ~ 1 사이의 값을 갖는 각 거리로써 일반적으로 벡터간의 유사도 값으로 간주하여 여러 용도로 사용됩니다. 해당 연산은 정의상 모든 대상 벡터와 연산을 해야하는 Brute Force 형태의 계산이 필요합니다. 이를 매우 효과적으로 근사하는 방법은 없기 떄문에 대규모 연산 혹은 서비스 시점에는 최근접 이웃 (ANN - Approximate Nearest Neighbor) 방식을 통하여 Recall을 희생하여 계산 성능을 올리는 방식이 Best Practice로 알려져 있습니다.

### ANN 방식엔 크게 다음과 같은 문제점들이 존재합니다.

- **유사도 계산 대상 벡터의 대한 초기 빌드 시간이 필요합니다.**
- **유사도 top k 를 추출할 경우 k 값이 일정 수준 이상일 경우 근사 성능이 감소합니다.**
- **벡터의 차원이 커질 경우 근사 성능이 크게 감소합니다. 즉 ANN을 사용할 경우 100 ~ 256 차원 이상이라면 적절한 차원 축소가 필요합니다.**
- **벡터의 숫자가 100,000 미만인 경우 ANN은 구조상 연산 성능에 큰 이득이 존재 하지 않습니다.**
- **즉 비교적 높은 차원의 벡터를 다루면서 많은 top k가 필요하거나 충분히 대상 벡터가 적다면 ANN의 활용도가 줄어듭니다.**

### 본 프로젝트에서는 다음과 같이 문제점을 해결합니다.

- **대상 벡터는 동적으로 모델 그래프를 생성하여 GPU의 상수 메모리로 고정합니다.**
- **메트릭스 연산으로 표현되는 내적 연산을 Tensorflow Graph의 텐서 연산을 통해서 동적 배치로 다수를 한번에 처리합니다.**
- **L2norm 연산과 Transpose 연산이 런타임에 불필요하게 일어나지 않도록 사전에 처리하여 저장하고 로딩합니다.**
- **akka-http, akka-stream 을 통하여 Dynamic Batch를 구현하고 Akka Http의 비동기 처리를 통하여 수 백 ~ 수 천 요청을 동시에 처리합니다.**

### 기존의 Best Practice와 SOTA 대비 다음과 같은 성능과 장점을 얻습니다.

- **http://ann-benchmarks.com 의 SOTA(ScaNN, 0.9876) 대비 Recal을 희생하지 않고 약 55 ~ 65% RPS(request per second)를 얻습니다.**
- **glove-100-angular 벤치마크 데이터 셋 기준 SOTA(ScaNN, 182초) 대비 2초 이내로 로딩 되며 배포시 5초 내외로 서버가 구동 됩니다.**
- **100,000 수준의 벡터의 경우 100 ~ 2048 차원에 대해서 4000 ~ 260 수준의 RPS(request per second)를 얻습니다.**
- **대상 벡터는 python의 numpy 형식을 통하여 npy 파일로 로드 할 수 있습니다.**
- **여러 환경으로 빌드된 tensorflow 런타임을 사용하기 때문에 linux, windows, mac등 여러 환경에서 쉽게 사용될 수 있습니다.**
- **비교적 작은 프로덕션 환경에서 Recall의 감소없이 throughput, latency를 고려하고 배포 파이프 라인을 단순화 하기 위한 목적으로 예제의 사용을 추천합니다.**

### 주의 사항 
- **ann-benchmarks 와 비교한 내용은 Recall 1 인 손실 없는 계산이며 배치 라이브러리 호출이 아닌 rest api의 end2end로 측정되었습니다.**
- **ann-benchmarks 와 비교는 공정한 비교가 아닙니다. ann-benchmarks는 AWS의 CPU r5.4xlarge 에서 측정 되었으며 현 예제의 GPU와는 환경과는 큰 차이가 있습니다.**
- **Dynamic Batch 상황시 cublas의 MatmulAlgoGetHeuristic의 동작에 의한 묵시적인 GEMM 알고리즘 변경으로 수치적 오차가 생길 가능성이 있습니다.**
- **GPU 메모리의 사양에 따라서 최대 가용 가능한 Dynamic Batch의 크기가 달라집니다. 일반적으로 메모리가 허용하는한 큰 값을 주는것이 더 높은 RPS 성능을 보입니다.**


## 기본 구성
- **최소한의 코드**, **최소한의 의존성**
- **Tensorflow-java-gpu** 를 Serving Runtime 으로 사용
- **akka-http** 를 통한 rest api 구성
- **akka-stream** 을 통한 dynamic batching 구현
- tensorflow-java 동적 그래프 생성을 통한 jvm gpu 연산 가속 및 serving


## docker
```
docker build . -f Dockerfile -t akka:0.1
docker run --gpus all -p 8080:8080 akka:0.1
```

## local build & run
```
sbt assembly

java -Dport=8080 \
-Dbatch=128 \
-Ddim=100 \
-Dsample=10000 \
-DnpyFile=./model/10000-100.npy \
-Dtimeout=10000 \
-DtakeSpinCountDelay=5 \
-Xmx8G \
-Xms8G \
-jar ./target/scala-2.12/akka-dynamic-batch-tensorflow-gpu-graph-cosine-similarity.jar

```
api
----------------------------------------------------------------
```
curl -X POST http://192.168.0.21:8080/cos -H "Content-Type: application/json"  -d '{"embedding":[0.0,1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.10,11.11,12.12,13.13,14.14,15.15,16.16,17.17,18.18,19.19,20.20,21.21,22.22,23.23,24.24,25.25,26.26,27.27,28.28,29.29,30.30,31.31,32.32,33.33,34.34,35.35,36.36,37.37,38.38,39.39,40.40,41.41,42.42,43.43,44.44,45.45,46.46,47.47,48.48,49.49,50.50,51.51,52.52,53.53,54.54,55.55,56.56,57.57,58.58,59.59,60.60,61.61,62.62,63.63,64.64,65.65,66.66,67.67,68.68,69.69,70.70,71.71,72.72,73.73,74.74,75.75,76.76,77.77,78.78,79.79,80.80,81.81,82.82,83.83,84.84,85.85,86.86,87.87,88.88,89.89,90.90,91.91,92.92,93.93,94.94,95.95,96.96,97.97,98.98,99.99]}'
```
```
{"result":[[33609,0.8780051],[9095,0.8760668],[165,0.8757745],[94124,0.8753646], ... ]}

{"result":[[index,score], ... ]}
```

Npy file
----------------------------------------------------------------
```
import numpy as np

item = 100000
dim = 100

a = np.random.rand(item, dim)
b = np.linalg.norm(a , axis=1, keepdims=True)
c = a / b
c = np.transpose(c).astype(np.float32).flatten()

print(c) #[0.07308075 0.00123816 0.17119586 ... 0.08993698 0.1488913  0.07554862
print(c.shape) #(1000000,)
print(c.dtype) #float32

np.save(f"./{item}-{dim}",c)
```

