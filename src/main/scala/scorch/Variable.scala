package scorch

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

case class Variable[@specialized(Float, Double) T: ClassTag](
    data: Tensor[T],
    f: Option[Function[T]] = None)(implicit ev: TensorNumeric[T]) {

  lazy val g: Tensor[T] = numsca.zerosLike(data)


}
