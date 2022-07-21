# Akka Dynamic Batch Tensorflow Serving Skeleton Api


## Spec
- **Java 11**
- **Scala 2.12.15**
- **SBT 1.6.2**
- **akka-actor-typed 2.6.8**
- **akka-stream 2.6.8**
- **akka-http 10.2.7**
- **akka-http-caching 10.2.7**
- **akka-http-spray-json 10.2.7**
- **tensorflow-core-platform-gpu 0.4.1**

## 기본 구성
- **최소한의 코드**, **최소한의 의존성**
- 4000 ~ 5000 RPS (requests per second) 수준의 Response performance
- **Tensorflow-java-gpu** 를 Serving Runtime 으로 사용 
- **akka-http** 를 통한 rest api 구성
- **akka-stream** 을 통한 dynamic batching 구현
- onnx-java-cpu, onnx-java-gpu Runtime 변경 가능
- tensorflow-java 동적 그래프 생성을 통한 jvm gpu 연산 가속 및 serving


## docker
```
docker build . -f Dockerfile -t akka:0.1
docker run -p 8080:8080 akka:0.1
```

## local build & run
```
sbt assembly

java -Dport=8080 \
-Dbatch=32 \
-Dtimeout=10000 \
-DtakeSpinCountDelay=5 \
-DmodelPath=./model \
-DvocabPath./model/vocab.txt \
-Xmx8G \
-Xms8G \
-jar ./target/scala-2.12/akka-dynamic-batch-tensorflow-gpu-bert.jar

```
api 
----------------------------------------------------------------
```
http://localhost:8080/bert?query=이 영화는 최고의 영화입니다

{
    result: [
        0.038097005,
        0.961903
    ]
}
```
```
http://localhost:8080/bert?query=최악이에요. 배우의 연기력도 좋지 않고 내용도 너무 허접합니다

{
    result: [
        0.99941707,
        0.00058293977
    ]
}
```

model
----------------------------------------------------------------
```
pip install transformers==4.20.1
python -m transformers.onnx --feature=sequence-classification --model=daekeun-ml/koelectra-small-v3-nsmc ./
```

```
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("model.onnx")  # load onnx model
output = prepare(onnx_model)
output.export_graph("model")
```

banchmark
----------------------------------------------------------------
```
koelectra-small 
https://huggingface.co/daekeun-ml/koelectra-small-v3-nsmc
Dynamic Batch Size : 196
Max Sequence Length : 32
RPS (requests per second):  4146.63
CPU : Ryzen 9 3950X
GPU : Nvidia Geforce rtx 3090 (495.29.05)
ENV : docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

```

```
196 threads and 196 connections
Thread Stats  Avg   Stdev   Max  +/- Stdev
Latency  47.08ms  5.25ms 169.83ms  77.96%
Req/Sec  21.16   4.69  30.00   77.00%
41672 requests in 10.05s, 6.64MB read
Requests/sec:  4146.63
Transfer/sec:  676.26KB
```
