package serving.http

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import spray.json.{DefaultJsonProtocol, RootJsonFormat}


trait JsonSupport extends SprayJsonSupport with DefaultJsonProtocol {

  implicit val bertResponseFormat: RootJsonFormat[resultBertResponse] =
    jsonFormat1(resultBertResponse)

}
