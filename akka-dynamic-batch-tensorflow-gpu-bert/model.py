#!pip install transformers==4.20.1
#!python -m transformers.onnx --feature=sequence-classification --model=daekeun-ml/koelectra-small-v3-nsmc ./
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("model.onnx")  # load onnx model
output = prepare(onnx_model)
output.export_graph("model")
