package serving.service


import akka.stream.QueueOfferResult
import serving.akka.AkkaManager._
import serving.cache.AkkaCacheManager.bertCache
import serving.config.ConfigManager

import serving.model.BertUtil

import scala.concurrent.Future

object BertService {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)
  private val TIMEOUT = ConfigManager.timeout

  def apply(query: String): Future[Option[Array[Float]]] = {

    val key: String = java.util.UUID.randomUUID().toString
    val token: Seq[Long] = BertUtil.tokenize(query)
    val queueSuccess: Future[QueueOfferResult] = AkkaQueueService.offer((key, token))

    val logitF: Future[Option[Array[Float]]] =
      bertCache.take(key, TIMEOUT).map(f => f.map(Option(_))).getOrElse(Future.successful(None))

    val probabilityF = logitF.map(logit => logit.map(BertUtil.softmax))

    probabilityF
  }
}
