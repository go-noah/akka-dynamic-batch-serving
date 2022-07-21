package serving.http

import akka.http.scaladsl.Http
import serving.akka.AkkaManager._
import serving.config.ConfigManager
import serving.service.BertService


object HttpServer extends JsonSupport {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)
  private val HTTP_PORT: Int = ConfigManager.port

  def main(args: Array[String]): Unit = {
    val bertRoute = new BertRoute()
    val routes = bertRoute.route
    log.info(s"Warmup start")
    BertService("warmup")
    val bindingFuture = Http().newServerAt("0.0.0.0", HTTP_PORT).bind(routes)
    bindingFuture.onComplete {
      case scala.util.Success(s) =>
        log.info(s"Server now online. $s")

      case scala.util.Failure(f) =>
        f.printStackTrace()
        log.error(s"Bind server error=${f.getMessage}")
    }
  }
}
