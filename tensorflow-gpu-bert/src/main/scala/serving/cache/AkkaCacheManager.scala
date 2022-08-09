package serving.cache

import scala.concurrent.duration._

object AkkaCacheManager {
  val bertCache: AkkaCache[Array[Float]] =
    new AkkaCache[Array[Float]]("BertCache", 100000, 10.minutes, 5.minutes)
}
