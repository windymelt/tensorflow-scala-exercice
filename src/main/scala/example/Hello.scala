package example

import org.platanios.tensorflow
import tensorflow.api.tensors.Tensor
import tensorflow.api.core.Shape

object Hello extends App with TF {
  val tensor = Tensor.zeros[Int](Shape(2, 5))
  println(tensor.summarize())
  manyTensors.foreach(t => println(t.summarize()))

  println("multiplying")
  println(multiply.summarize())

  println("ones")
  println(ones.summarize())

  slice

  sess
}

trait TF {
  def manyTensors: Seq[Tensor[Any]] = {
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
    Tensor.ones[Int](Shape(3, 3))
  }

  def slice: Unit = {
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
}
