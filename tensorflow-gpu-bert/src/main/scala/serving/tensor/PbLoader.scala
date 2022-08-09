package serving.tensor

import org.tensorflow.{SavedModelBundle, Tensor}

case class InputTensor(inputLayerNames: String,
                       tensor: Tensor)

object PbLoader {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)

  def apply(pbName: String, tags: String = "serve"): TensorFlowFunctionProvider = {
    try {
      val model = SavedModelBundle.load(pbName, tags)
      val tensorFlowFunctionProvider = new TensorFlowFunctionProvider(model)

      log.info(s"modelInfo=${tensorFlowFunctionProvider.modelInfo()}")

      tensorFlowFunctionProvider
    } catch {
      case e: NoSuchElementException =>
        log.warn(s"load error=${e.getMessage}" +
          s"model=$pbName, " +
          s"tags=$tags")
        throw e
    }
  }

}
