package serving.config

import com.typesafe.config.ConfigFactory

object ConfigManager {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)
  private val config = ConfigFactory.load
  val port: Int =  System.getProperty("port","8080").toInt
  val batch: Int =  System.getProperty("batch","32").toInt
  val timeout: Int =  System.getProperty("timeout","10000").toInt
  val takeSpinCountDelay : Int =  System.getProperty("takeSpinCountDelay","5").toInt

  val bertPath = System.getProperty("modelPath","./model")
  val vocabPath = System.getProperty("vocabPath","./model/vocab.txt")

  log.info("Load config. " +
    s"batch=$batch, " +
    s"timeout=$timeout, " +
    s"takeSpinCountDelay=$takeSpinCountDelay"
  )

}
