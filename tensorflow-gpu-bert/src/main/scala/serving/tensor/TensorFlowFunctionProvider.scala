package serving.tensor

import org.tensorflow.{SavedModelBundle, Tensor}

import scala.collection.JavaConverters._
import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`

class TensorFlowFunctionProvider(model: SavedModelBundle) extends AutoCloseable {
  private val savedModelBundle: SavedModelBundle = model
  private val function = model.function("serving_default")

  def run(inputTensors: Seq[InputTensor]): Map[String, Tensor] = {
    val input = inputTensors.map(x => (x.inputLayerNames, x.tensor)).toMap.asJava
    val runner = function.call(input)

    val resultTensors: Map[String, Tensor] = runner.asScala.map(x => x._1 -> x._2).toMap

    inputTensors.foreach(_.tensor.close())
    resultTensors
  }

  def modelInfo(signatureKey: String = "serving_default"): String = {
    val fnc = savedModelBundle.function("serving_default")

    val inputNames = fnc.signature().inputNames()
    val outputNames = fnc.signature().outputNames()

    val inputOps = inputNames.map { x =>
      val key = fnc.signature().getInputs.get(x)

      Map(x -> (key, key.dataType, key.shape)).mkString
    }.mkString(", ")

    val outputOps = outputNames.map { x =>
      val key = fnc.signature().getOutputs.get(x)

      Map(x -> (key.name, key.dataType, key.shape)).mkString
    }.mkString(", ")

    Map("signatureKey" -> signatureKey,
      "inputNames" -> inputNames,
      "inputOps" -> inputOps,
      "outputNames" -> outputNames,
      "outputOps" -> outputOps).mkString
  }

  override def close(): Unit = {
    model.close()
  }
}

