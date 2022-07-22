package serving.model

import ai.onnxruntime.{OnnxTensor, OrtEnvironment}
import org.json4s.DefaultFormats
import serving.config.ConfigManager
import serving.onnx.OnnxProvider

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

    val inputs = input.flatMap(_.padTo(maxInputLength, 1L))
    val masks = input.flatMap(x => Array.fill[Long](x.length)(1L).padTo(maxInputLength, 0L))
    val type_ids = Array.fill[Long](batchSize * maxInputLength)(0L)

    val inputs_buf = LongBuffer.wrap(inputs)
    val masks_buf = LongBuffer.wrap(masks)
    val type_ids_buf = LongBuffer.wrap(type_ids)

    val inputIds: OnnxTensor =
      OnnxTensor.createTensor(OrtEnvironment.getEnvironment, inputs_buf, Array(batchSize.toLong, maxInputLength.toLong))
    val tokenTypeIds: OnnxTensor =
      OnnxTensor.createTensor(OrtEnvironment.getEnvironment, type_ids_buf, Array(batchSize.toLong, maxInputLength.toLong))
    val attentionMask: OnnxTensor =
      OnnxTensor.createTensor(OrtEnvironment.getEnvironment, masks_buf, Array(batchSize.toLong, maxInputLength.toLong))

    val onnxInputs: Map[String, OnnxTensor] =
      Map("input_ids" -> inputIds, "token_type_ids" -> tokenTypeIds, "attention_mask" -> attentionMask)

    val run: Iterable[AnyRef] = onnxProvider.run(onnxInputs).map(x => x.getValue.getValue)

    val result: Iterable[Array[Float]] =
      run.flatMap(x => x.asInstanceOf[Array[Any]].map(s => s.asInstanceOf[Array[Float]]))

    result

  }

  def apply(inputs: Array[Seq[Long]]): Iterable[Array[Float]] = {
    run(inputs)
  }
}

