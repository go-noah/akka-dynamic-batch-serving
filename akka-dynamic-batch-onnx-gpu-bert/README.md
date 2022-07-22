# Akka Dynamic Batch Onnx Serving Skeleton Api


## Spec
- **Java 11 (Java 1.8 Support)**
- **Scala 2.12.15**
- **SBT 1.6.2**
- **akka-actor-typed 2.6.8**
- **akka-stream 2.6.8**
- **akka-http 10.2.7**
- **akka-http-caching 10.2.7**
- **akka-http-spray-json 10.2.7**
- **onnxruntime_gpu 1.12.0**

## 기본 구성
- **최소한의 코드**, **최소한의 의존성**
- 4000 ~ 20000 RPS (requests per second) 수준의 Response performance
- **onnxruntime_gpu** 를 Serving Runtime 으로 사용 
- **akka-http** 를 통한 rest api 구성
- **akka-stream** 을 통한 dynamic batching 구현
- tensorflow-java-cpu, tensorflow-java-gpu Runtime 변경 가능

## docker
```
docker build . -f Dockerfile -t akka:0.1
docker run -p 8080:8080 akka:0.1
```

## local build & run
```
sbt assembly

java -Dport=8080 \
-Dbatch=256 \
-Dtimeout=10000 \
-DtakeSpinCountDelay=5 \
-DmodelPath=./model/model.onnx \
-DvocabPath./model/vocab.txt \
-Xmx8G \
-Xms8G \
-jar ./target/scala-2.12/akka-dynamic-batch-onnx-gpu-bert.jar

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

banchmark
----------------------------------------------------------------
```
CPU : Ryzen 9 3950X
GPU : Nvidia Geforce rtx 3090 (495.29.05)
ENV : docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04
CMD : hey -n 100000 -c 256 http://servicehost:8080/bert?query=이 영화는 최고의 영화입니다
```

```
koelectra-small 
https://huggingface.co/daekeun-ml/koelectra-small-v3-nsmc 
```

| dynamic batch    |     8 |    16 |    32 |   64 |  128 |  256 |   512 |
|------------------|-------|-------|-------|------|------|------|-------|
| Requests/sec     | **19035** | 14434 | 10400 | 6792 | 4132 | 1912 |   788 |
| latency avg (ms) |    13 |    17 |  23.6 | 37.2 | 61.5 | 71.1 | 161.4 |

```
bert-base
https://huggingface.co/sangrimlee/bert-base-multilingual-cased-nsmc
```

| dynamic batch    |    8 |   16 |   32 |    64 |   128 |   256 |    512 |
|-----------------:|-----:|-----:|-----:|------:|------:|------:|-------:|
| Requests/sec     | 8608 | 5595 | 3314 |  **1799** |   904 |   429 |    187 |
| latency avg (ms) |   27 | 44.6 | 76.1 | 144.4 | 280.8 | 587.1 | 1340.3 |
