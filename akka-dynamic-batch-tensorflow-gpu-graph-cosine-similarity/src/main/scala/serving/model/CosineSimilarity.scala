package serving.model

import org.jetbrains.bio.npy.NpyFile
import org.json4s.DefaultFormats
import org.tensorflow.{Graph, Tensor}
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.op.Ops
import org.tensorflow.op.core.{Constant, Placeholder}
import org.tensorflow.types.TFloat32
import serving.config.ConfigManager
import serving.tensor.TensorFlowProvider
import serving.tensor.InputTensor


import java.nio.file.Paths

final case class Vec(vec: Seq[(String, Array[Float])])

object CosineSimilarity {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)
  implicit val formats: DefaultFormats = DefaultFormats

  val wLength: Int = ConfigManager.sample
  val dim: Int = ConfigManager.dim

  log.info(s"CosineSimilarity Initialize start")

  val dataVectorNumpyArray: Array[Float] = {
    val npyFilePath = ConfigManager.npyFile
    log.info(s"Npy file load = ${npyFilePath}")
    NpyFile.read(Paths.get(npyFilePath),Int.MaxValue).asFloatArray()
  }

  def model(k: Int = 100): Graph = {

    val wArray: Array[Float] = dataVectorNumpyArray

    assert(wArray.length == (dim * wLength))

    val graph = new Graph()
    val tf = Ops.create(graph)

    val wTensor: Constant[TFloat32] = {
      val fp32 = DataBuffers.ofFloats(wLength * dim).write(wArray,0,wLength * dim)
      tf.constant(Shape.of(1, dim, wLength), fp32)
    }

    val v = tf.withName("input").placeholder(classOf[TFloat32], Placeholder.shape(Shape.of(-1, dim, 1)))

    val mul = tf.math.mul(v, wTensor)
    val cosineSimilarity = tf.reduceSum(mul, tf.array(1))
    val topK = tf.withName("output").nn.topK(cosineSimilarity, tf.constant(k))

    graph

  }

  private val modelProvider: TensorFlowProvider = new TensorFlowProvider(model())


  def run(v: Array[Array[Float]], dim: Int = dim, k: Int = 100): Array[Array[(Int, Float)]] = {
    val startTime = System.currentTimeMillis()

    val vLength: Int = v.length
    val vTensor: TFloat32 = Tensor.of(classOf[TFloat32], Shape.of(vLength, dim, 1))
    val vArray: Array[Float] = v.flatten

    DataBuffers.ofFloats(vLength * dim).write(vArray)
      .copyTo(vTensor.asRawTensor().data().asFloats(), vLength * dim)

    val inputTensors = Seq(
      InputTensor("input", vTensor)
    )

    val outputTensors: Seq[(String, Int)] = Seq(
      ("output", 0),
      ("output", 1)
    )

    val resultTensors = modelProvider.run(inputTensors, outputTensors)

    val scores = resultTensors.get(outputTensors.head)
      .map(x => {
        val tensor = x
        val size = tensor.shape().asArray().product.toInt
        val array = Array.ofDim[Float](size)
        tensor.asRawTensor().data().asFloats().read(array)
        array.grouped(k)
      }).head

    val index = resultTensors.get(outputTensors.last)
      .map(x => {
        val tensor = x
        val size = tensor.shape().asArray().product.toInt
        val array = Array.ofDim[Int](size)
        tensor.asRawTensor().data().asInts().read(array)
        array.grouped(k)
      }).head

    log.debug(s"cosineSimilarity elapsedTime=${System.currentTimeMillis() - startTime}ms")
    (index zip scores).map(x => x._1 zip x._2).map(y => y).toArray
  }

  def apply(v: Array[Array[Float]]): Array[Array[(Int, Float)]] = {
    run(v)
  }

}
