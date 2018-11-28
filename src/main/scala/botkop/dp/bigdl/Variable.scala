package botkop.dp.bigdl

import com.intel.analytics.bigdl.tensor.Tensor

case class Variable[T](data: Tensor[T],
                       f: Option[Function] = None,
                       name: Option[String] = None) {

//  lazy val g: Tensor[T] = Tensor[T](data.size())

}

