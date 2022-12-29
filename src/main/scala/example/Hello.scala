package example

import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.core.Shape

object Hello extends App {
  val tensor = Tensor.zeros[Int](Shape(2, 5))
  println(tensor.summarize())
}

