package serving.model

import ai.onnxruntime.{OnnxTensor, OrtEnvironment}
import org.json4s.DefaultFormats
import serving.config.ConfigManager
import serving.onnx
import serving.onnx.{InputOnnxTensor, OnnxProvider}

import java.nio.LongBuffer

object BertSmallNsmc {

  implicit val formats: DefaultFormats = DefaultFormats
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)

  log.info(s"BertSmallNsmc initialize start")

  private val INITIALIZE_START_TIME = System.currentTimeMillis()
  private val ONNX_PATH = ConfigManager.modelPath
  private val onnxProvider = new OnnxProvider(ONNX_PATH)

  log.info(s"BertSmallNsmc initialize done. elapsedTime=${System.currentTimeMillis() - INITIALIZE_START_TIME}ms")

  def run(input: Array[Seq[Long]]): Iterable[Array[Float]] = {

    val maxInputLength = input.map(x => x.length).max
    val batchSize = input.length

    val inputArray = input.flatMap(_.padTo(maxInputLength, 1L))
    val typeArray = Array.fill[Long](batchSize * maxInputLength)(0L)
    val maskArray = input.flatMap(x => Array.fill[Long](x.length)(1L).padTo(maxInputLength, 0L))

    val inputBuf = LongBuffer.wrap(inputArray)
    val typesBuf = LongBuffer.wrap(typeArray)
    val maskBuf = LongBuffer.wrap(maskArray)

    val inputIds: OnnxTensor =
      OnnxTensor.createTensor(OrtEnvironment.getEnvironment, inputBuf, Array(batchSize.toLong, maxInputLength.toLong))
    val typeIds: OnnxTensor =
      OnnxTensor.createTensor(OrtEnvironment.getEnvironment, typesBuf, Array(batchSize.toLong, maxInputLength.toLong))
    val attentionMasks: OnnxTensor =
      OnnxTensor.createTensor(OrtEnvironment.getEnvironment, maskBuf, Array(batchSize.toLong, maxInputLength.toLong))

    val inputOnnxTensors: Seq[InputOnnxTensor] = Seq(
      InputOnnxTensor("input_ids", inputIds),
      InputOnnxTensor("token_type_ids", typeIds),
      InputOnnxTensor("attention_mask", attentionMasks),
    )

    val result: Iterable[Array[Float]] = onnxProvider.run(inputOnnxTensors)
      .flatMap(x => x.asInstanceOf[Array[Any]].map(s => s.asInstanceOf[Array[Float]]))
    result

  }

  def apply(inputs: Array[Seq[Long]]): Iterable[Array[Float]] = {
    run(inputs)
  }
}

