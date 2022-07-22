package serving.onnx

import ai.onnxruntime.{OnnxTensor, OnnxValue, OrtEnvironment, OrtSession}

import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`
import java.nio.LongBuffer
import java.nio.file.{Files, Paths}
import java.util
import collection.JavaConverters._
import scala.collection.JavaConversions.mapAsJavaMap


case class InputOnnxTensor(inputLayerNames: String,
                           tensor: OnnxTensor)

class OnnxProvider(modelPath: String) {

  val env: OrtEnvironment = OrtEnvironment.getEnvironment()

  val gpuDeviceId = 0

  var sessionOptions: OrtSession.SessionOptions = {
    val option = new OrtSession.SessionOptions()
    option.addCUDA(gpuDeviceId)
    option
  }

  val session: OrtSession = env.createSession(modelPath, sessionOptions)

  def run(input: Seq[InputOnnxTensor]): Iterable[(String, OnnxValue)] = {

    val inputs: util.Map[String, OnnxTensor] = mapAsJavaMap(input.map(x => x.inputLayerNames -> x.tensor).toMap)
    val results: Iterable[(String, OnnxValue)] = session.run(inputs).asScala.map(x => x.getKey -> x.getValue)

    results

  }
}

