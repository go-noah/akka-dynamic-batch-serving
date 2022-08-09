package serving.http

import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.StatusCodes.ServiceUnavailable
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.{Route, StandardRoute}
import serving.akka.AkkaManager._
import serving.service.BertService

import scala.concurrent.Future
import scala.language.postfixOps
import scala.util.{Failure, Success}

final case class resultBertResponse(result: Array[Float])

class BertRoute extends JsonSupport {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)

  def serviceUnavailable: StandardRoute =
    complete(
      HttpResponse(ServiceUnavailable,
        entity = "The server is currently unavailable (because it is overloaded or down for maintenance).")
    )

  val route: Route = bertRoute()

  private def bertRoute(): Route =
    pathPrefix("bert") {
      parameters('query.as[String]) { query =>

        val probabilityF: Future[Option[Array[Float]]] = BertService(query)

        val resultF: Future[StandardRoute] = probabilityF.map {
          case Some(prob) =>
            complete(resultBertResponse(prob))
          case None =>
            log.error(s"503 serviceUnavailable : Empty result")
            serviceUnavailable
        }

        onComplete(resultF) {
          case Success(success) =>
            success
          case Failure(f) =>
            log.error(s"503 serviceUnavailable : $f")
            serviceUnavailable
        }

      }
    }
}
