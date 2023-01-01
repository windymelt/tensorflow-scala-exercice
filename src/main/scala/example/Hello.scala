package example

import org.platanios.tensorflow
import java.nio.file.Paths

object Hello extends App with TF {
  import tensorflow.api.tensors.Tensor
  import tensorflow.api.core.Shape

  val tensor = Tensor.zeros[Int](Shape(2, 5))
  println(tensor.summarize())
  manyTensors.foreach(t => println(t.summarize()))

  println("multiplying")
  println(multiply.summarize())

  println("ones")
  println(ones.summarize())

  slice

  sess

  //learn
  rabbit
}

trait TF {
  import tensorflow.api.tensors.Tensor
  def manyTensors: Seq[Tensor[Any]] = {
    import tensorflow.api.core.Shape
    val a = Tensor[Int](1, 2)                  // Creates a Tensor[Int] with shape [2]
    val b = Tensor[Long](1L, 2)                // Creates a Tensor[Long] with shape [2]
    val c = Tensor[Float](3.0f)                // Creates a Tensor[Float] with shape [1]
    val d = Tensor[Double](-4.0)               // Creates a Tensor[Double] with shape [1]
    val e = Tensor.empty[Int]                  // Creates an empty Tensor[Int] with shape [0]
    val z = Tensor.zeros[Float](Shape(5, 2))   // Creates a zeros Tensor[Float] with shape [5, 2]
    val r = Tensor.randn(Double, Shape(10, 3)) // Creates a Tensor[Double] with shape [10, 3] and
                                               // elements drawn from the standard Normal distribution.
    Seq(a, b, c, d, e, z, r)
  }

  def multiply: Tensor[Double] = {
    val x = Tensor[Double](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    x * x.transpose()
  }

  def ones: Tensor[Int] = {
    import tensorflow.api.core.Shape
    Tensor.ones[Int](Shape(3, 3))
  }

  def slice: Unit = {
    import tensorflow.api.core.Shape
    import tensorflow.api.{---, ::}
    val t = Tensor.zeros[Int](Shape(4, 2, 3, 8)) // 4 x 2 x 3 x 8 の4次元テンソルを0で初期化する
    val t1 = t(::, ::, 1, ::) // 3次元目を1にインデキシングする
    println(t1.summarize())
    val t3 = Tensor.zeros[Int](Shape(5))
    import tensorflow.api._ // n :: m の記法のために必要
    val t2 = t(1 :: -2, ---, 2)
    println(t2.summarize())
    println(t3(::, NewAxis, NewAxis).summarize())
  }

  def sess: Unit = {
    println("session")
    import tensorflow.api.core.client.Session
    val sess = Session()
    val o1 = Tensor[Int](Seq(Seq(1, 2), Seq(3, 4))).toOutput
    val o2 = Tensor[Int](Seq(Seq(1, 2), Seq(3, 4))).toOutput
    val o1o2 = o1 * o2
    val t1t2 = sess.run(fetches = Seq(o1o2))
    println(t1t2.summarize())
    sess.close()
  }

  def learn: Unit = {
    import tensorflow.api._
    import tensorflow.api.learn.Model
    import tensorflow.api.tensors.Tensor

    val trainDS = Tensor[Float](Seq(0, 0), Seq(0, 1), Seq(1, 0), Seq(1, 1))
    println(trainDS.summarize())
    val trainLabels = Tensor[Float](Seq(0, 1, 1, 0)).transpose()
    println(trainLabels.summarize())

    import tensorflow.api.learn.layers._
    val input = Input(FLOAT32, Shape(4, 2))
    val trainInput = Input(FLOAT32, Shape(4, 1))

    val layer = Linear[Float]("inputLinear", 2, useBias = true) >> ReLU[Float]("hidden ReLU") >> Linear[Float]("outputLinear", 1)

    val loss = L2Loss[Float, Float]("l2loss") >> Mean("loss/mean") >> ScalarSummary(name = "Loss", tag = "Loss") // 本当はMSEが欲しいんだけど・・・

    val optimizer = tensorflow.api.ops.training.optimizers.AdaGrad(0.01f)

    val model = Model.simpleSupervised(input, trainInput, layer, loss, optimizer)

    val summariesDir = Paths.get("/tmp/summaries")
    val estimator = tensorflow.api.learn.estimators.InMemoryEstimator(
      model,
      configurationBase = tensorflow.api.learn.Configuration(Some(summariesDir)),
      trainHooks = Set(
        tensorflow.api.learn.hooks.SummarySaver(summariesDir, tensorflow.api.learn.hooks.StepHookTrigger(100)),
        tensorflow.api.learn.hooks.CheckpointSaver(summariesDir, tensorflow.api.learn.hooks.StepHookTrigger(1000))
      ),
      tensorBoardConfig = tensorflow.api.config.TensorBoardConfig(summariesDir),
    )

    val trainDataSet = tensorflow.api.ops.data.Data.datasetFromTensors(trainDS)
    val trainLabelsDataSet = tensorflow.api.ops.data.Data.datasetFromTensors(trainLabels)
    val trainData = trainDataSet.zip(trainLabelsDataSet).repeat().shuffle(10000).prefetch(10)
    estimator.train(() => trainData, tensorflow.api.learn.StopCriteria(maxSteps = Some(500000L)))

  }

  def rabbit: Unit = {
    // ウサギ検出器
    // CIFAR-100の65(ウサギ)を使ってウサギを判定できるようにする
    import tensorflow.data.image.CIFARLoader
    val dataSet = CIFARLoader.load(Paths.get("/home/windymelt/Downloads/cifar-100-python"), CIFARLoader.CIFAR_100)

    import tensorflow.api.ops.data.Data
    import tensorflow.api.::
    val trainImages = () => Data.datasetFromTensorSlices(dataSet.trainImages, "TrainImages").map(_.toFloat)

    // super-categoryは捨ててcategoryだけ得るために(::, 1)で変形する
    val trainLabels = () => Data.datasetFromTensorSlices(dataSet.trainLabels(::, 1), "TrainLabels").map(_.toLong)
    val trainData = () => trainImages().zip(trainLabels())
      .repeat()
      .shuffle(10000)
      .batch(32)
      .prefetch(10)

    import tensorflow.api._
    import tensorflow.api.learn.layers._
    val input = Input(
      FLOAT32,
      Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2), dataSet.trainImages.shape(3))
    )
    val trainInput = Input(INT64, Shape(-1))

    import tensorflow.api.ops.NN.SameConvPadding
    val layer = Conv2D[Float]("Layer_0/Conv2D", Shape(2, 2, 3, 16), 1, 1, SameConvPadding) >>
        AddBias[Float]("Layer_0/Bias") >>
        ReLU[Float]("Layer_0/ReLU", 0.1f) >>
        MaxPool[Float]("Layer_0/MaxPool", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
        Conv2D[Float]("Layer_1/Conv2D", Shape(2, 2, 16, 32), 1, 1, SameConvPadding) >>
        AddBias[Float]("Bias_1") >>
        ReLU[Float]("Layer_1/ReLU", 0.1f) >>
        MaxPool[Float]("Layer_1/MaxPool", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
        Flatten[Float]("Layer_2/Flatten") >>
        Linear[Float]("Layer_2/Linear", 256) >>
        ReLU[Float]("Layer_2/ReLU", 0.1f) >>
    Linear[Float]("OutputLayer/Linear", 100)

    val loss = SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss/CrossEntropy") >>
    Mean[Float]("Loss/Mean") >>
    ScalarSummary[Float]("Loss/Summary", "Loss")

    val optimizer = tf.train.AdaGrad(0.1f)

    val model = tf.learn.Model.simpleSupervised(
      input = input,
      trainInput = trainInput,
      layer = layer,
      loss = loss,
      optimizer = optimizer)

    val summariesDir = Paths.get("temp/cnn-cifar")

    val estimator = tensorflow.api.learn.estimators.InMemoryEstimator(
      model,
      tensorflow.api.learn.Configuration(Some(summariesDir)),
      tensorflow.api.learn.StopCriteria(maxSteps = Some(100000)),
      Set(
        tensorflow.api.learn.hooks.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        tensorflow.api.learn.hooks.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tensorflow.api.learn.hooks.StepHookTrigger(100)),
        tensorflow.api.learn.hooks.CheckpointSaver(summariesDir, tensorflow.api.learn.hooks.StepHookTrigger(1000))),
      tensorBoardConfig = tensorflow.api.config.TensorBoardConfig(summariesDir, reloadInterval = 1))

    estimator.train(trainData, tensorflow.api.learn.StopCriteria(maxSteps = Some(1000)))
  }
}
