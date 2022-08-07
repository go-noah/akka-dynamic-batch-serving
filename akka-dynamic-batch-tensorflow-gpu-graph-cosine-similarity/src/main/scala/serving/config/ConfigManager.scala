package serving.config

import com.typesafe.config.ConfigFactory

object ConfigManager {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)
  private val config = ConfigFactory.load
  val port: Int =  System.getProperty("port","8000").toInt
  val timeout: Int =  System.getProperty("timeout","10000").toInt
  val takeSpinCountDelay : Int =  System.getProperty("takeSpinCountDelay","5").toInt

  val dim: Int = System.getProperty("dim","100").toInt
  val sample: Int =  System.getProperty("sample","10000").toInt
  val batch: Int =  System.getProperty("batch","16").toInt
  val npyFile: String = System.getProperty("npyFile", "./model/10000-100.npy")

  log.info("Load config. " +
    s"port=$port, " +
    s"batch=$batch, " +
    s"timeout=$timeout, " +
    s"takeSpinCountDelay=$takeSpinCountDelay" +
    s"dim=$dim" +
    s"sample=$sample" +
    s"batch=$batch" +
    s"npyFile=$npyFile"
  )

}