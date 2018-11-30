import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

package object numsca {

  def zeros[@specialized(Float, Double) T: ClassTag](shape: Int*)(
      implicit ev: TensorNumeric[T]): Tensor[T] = Tensor.apply(shape: _*)

  def zerosLike[@specialized(Float, Double) T: ClassTag](t: Tensor[T])(
      implicit ev: TensorNumeric[T]): Tensor[T] = Tensor.apply(t.size())

  def randn[@specialized(Float, Double) T: ClassTag](shape: Int*)(
      implicit ev: TensorNumeric[T]): Tensor[T] =
    Tensor.apply(shape: _*).randn()

  def floatDomain[R](block: => R): R = {
    implicit val ev: TensorNumeric[Float] = NumericFloat
    block
  }

  trait FloatDomain {
    implicit val ev: TensorNumeric[Float] = NumericFloat
  }

  trait Domain[T] {
    implicit val ev: TensorNumeric[T]
  }

}
