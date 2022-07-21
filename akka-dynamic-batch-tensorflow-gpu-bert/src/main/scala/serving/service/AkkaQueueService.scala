package serving.service

import akka.stream.scaladsl.{Keep, Sink, Source, SourceQueueWithComplete}
import akka.stream.{OverflowStrategy, QueueOfferResult}
import serving.akka.AkkaManager._
import serving.cache.AkkaCacheManager.bertCache
import serving.config.ConfigManager
import serving.model.BertSmallNsmc

import scala.concurrent.Future
import scala.concurrent.duration.DurationInt

object AkkaQueueService {
  private val log = org.slf4j.LoggerFactory.getLogger(this.getClass)

  private type K = String
  private type V = Seq[Long]

  private val BUFFER_SIZE: Int = 100000
  private val PROCESS_SIZE: Int = ConfigManager.batch


  private val queue: SourceQueueWithComplete[(K, V)] =
    Source.queue[(K, V)](BUFFER_SIZE, OverflowStrategy.backpressure)
      .groupedWithin(PROCESS_SIZE, 5.millis)
      .toMat(Sink.foreach((x: Seq[(K, V)]) => betch(x)))(Keep.left)
      .run()

  private def betch(inputs: Seq[(K, V)]): Seq[String] = {
    val startTime = System.currentTimeMillis()
    val request = inputs.map(_._2).toArray
    val key = inputs.map(x => x._1)
    val value = BertSmallNsmc(request)
    log.info(s"akka dynamic batching size = ${inputs.length} elapsedTime=${System.currentTimeMillis() - startTime}ms")
    (key zip value).foreach(x => bertCache.put(key = x._1, value = x._2))
    key
  }

  def offer(x: (K, V)): Future[QueueOfferResult] = {
    queue.offer(x)
  }

}
