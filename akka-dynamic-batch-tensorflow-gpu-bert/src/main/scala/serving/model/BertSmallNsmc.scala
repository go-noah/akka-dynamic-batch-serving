package serving.model

import org.json4s.DefaultFormats
import org.tensorflow.Tensor
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TInt64
import serving.config.ConfigManager
import serving.tensor.{InputTensor, PbLoader, TensorFlowFunctionProvider}

object BertSmallNsmc {

  implicit val formats: DefaultFormats = DefaultFormats
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)

  log.info(s"BertSmallNsmc initialize start")

  private val INITIALIZE_START_TIME = System.currentTimeMillis()
  private val BERT_PATH = ConfigManager.bertPath
  private val modelProvider: TensorFlowFunctionProvider = PbLoader(BERT_PATH)

  log.info(s"BertSmallNsmc initialize done. elapsedTime=${System.currentTimeMillis() - INITIALIZE_START_TIME}ms")


  def longTensorOf(input: Seq[Long], shape: Shape): TInt64 = {
    val tensor = Tensor.of(classOf[TInt64], shape)
    val size = shape.asArray().product
    DataBuffers.ofLongs(size).write(input.toArray).copyTo(tensor.asRawTensor().data().asLongs(), size)
    tensor
  }


  def run(modelProvider: TensorFlowFunctionProvider,
          input: Seq[Seq[Long]]): Iterator[Array[Float]] = {

    val startTime = System.currentTimeMillis()
    val maxInputLength = input.map(x => x.length).max
    val batchSize = input.length
    val shape = Shape.of(batchSize, maxInputLength)
    val size =  batchSize * maxInputLength

    val tokenIds: TInt64 = longTensorOf(input.flatMap(_.padTo(maxInputLength, 0L)), shape)

    val attentionMask: TInt64 = longTensorOf(
      input.flatMap(x => Array.fill[Long](x.length)(1L).padTo(maxInputLength, 0L)), shape)

    val tokenTypeIds: TInt64 = longTensorOf(Array.fill[Long](size)(0L), shape)

    val inputTensors: Seq[InputTensor] = Seq(
      InputTensor("input_ids", tokenIds),
      InputTensor("attention_mask", attentionMask),
      InputTensor("token_type_ids", tokenTypeIds),
    )

    val resultTensors: Map[String, Tensor] = modelProvider.run(inputTensors)

    val result: Iterator[Array[Float]] = resultTensors.get("logits").map(
      tensor => {
        val size = tensor.shape().asArray() //Array(-1,2)
        val arr = Array.ofDim[Float](size.product.toInt)
        tensor.asRawTensor().data().asFloats().read(arr)
        arr.grouped(size.last.toInt)
      }).get

    log.debug(s"elapsedTime=${System.currentTimeMillis() - startTime}ms")

    result
  }


  def apply(inputs: Array[Seq[Long]]): Array[Array[Float]] = {
    run(modelProvider, inputs).toArray
  }
}

