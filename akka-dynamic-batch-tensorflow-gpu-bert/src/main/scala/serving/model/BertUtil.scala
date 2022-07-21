package serving.model

import com.robrua.nlp.bert.FullTokenizer
import org.json4s.DefaultFormats
import serving.config.ConfigManager

import java.nio.file.Paths

object BertUtil {

  implicit val formats: DefaultFormats = DefaultFormats
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)

  log.info(s"Initialize start")

  private val INITIALIZE_START_TIME = System.currentTimeMillis()
  private val VOCAB_PATH = ConfigManager.vocabPath
  private val Tokenizer = new FullTokenizer(Paths.get(VOCAB_PATH))
  private val tokenMap = {
    val file = scala.io.Source.fromFile(VOCAB_PATH)
    val indexMap = file.getLines.zipWithIndex.toMap
    file.close()
    indexMap
  }

  log.info(s"BertUtil initialize done. elapsedTime=${System.currentTimeMillis() - INITIALIZE_START_TIME}ms")


  def softmax(arr: Array[Float]): Array[Float] = {
    val exp = arr.map(scala.math.exp(_))
    val sum = exp.sum
    exp.map(x => (x / sum).toFloat)
  }

  def tokenize (input : String): Seq[Long] = {
    val token = Tokenizer.tokenize(input).map(x => tokenMap(x).toLong).toSeq
    Seq(2L) ++ token ++ Seq(3L)
  }

}

