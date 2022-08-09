package serving.service

import akka.stream.QueueOfferResult
import serving.akka.AkkaManager._
import serving.config.ConfigManager
import serving.cache.AkkaCacheManager.CosineCache
import serving.config.ConfigManager
import serving.http.Embedding

import scala.concurrent.Future

object CosineSimilarityService {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)
  private val timeout = ConfigManager.timeout

  def apply(embedding: Embedding): Future[Option[Array[(Int, Float)]]] = {
    val emb: Array[Float] = embedding.embedding.padTo(ConfigManager.dim, 1.0f).slice(0, ConfigManager.dim)
    val l2NormEmb: Array[Float] = l2Norm(emb)
    val key: String = java.util.UUID.randomUUID().toString

    val queueSuccess: Future[QueueOfferResult] = AkkaQueueService.offer((key, l2NormEmb))

    val scoreF: Future[Option[Array[(Int, Float)]]] =
      CosineCache.take(key, timeout).map(f => f.map(Option(_))).getOrElse(Future.successful(None))
    scoreF
  }

  def l2Norm(v: Array[Float]): Array[Float] = {
    val l2_norm = math.sqrt(v.map(i => i * i).sum).toFloat
    v.map(_ / l2_norm)
  }

}
