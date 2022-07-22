package serving.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import java.util
import collection.JavaConverters._
import scala.collection.JavaConversions.mapAsJavaMap
import ai.onnxruntime.{OnnxTensor, OnnxValue}

case class InputOnnxTensor(inputLayerNames: String,
                           tensor: Array[Float])

class OnnxProvider(modelPath: String) {

  val env: OrtEnvironment = OrtEnvironment.getEnvironment()

  val gpuDeviceId = 0

  var sessionOptions: OrtSession.SessionOptions = {
    val option = new OrtSession.SessionOptions()
    option.addCUDA(gpuDeviceId)
    option
  }

  val session: OrtSession = env.createSession(modelPath, sessionOptions)

  def run(onnxInputs: scala.collection.immutable.Map[String, OnnxTensor]): Iterable[util.Map.Entry[String, OnnxValue]] = {
    val results: OrtSession.Result = session.run(onnxInputs)
    results.asScala
  }
}

