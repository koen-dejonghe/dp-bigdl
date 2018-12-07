import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.tensor._

package object ns {

  def zeros(shape: Int*): Ndarray = Ndarray(Tensor(shape: _*))
  def zerosLike(t: Ndarray): Ndarray = Ndarray(Tensor(t.shape.toArray))
  def randn(shape: Int*): Ndarray = Ndarray(Tensor(shape: _*).randn())
  def arange(stop: Int): Ndarray = arange(0, stop)
  def arange(start: Int, stop: Int): Ndarray = {
    val array = Array.range(start, stop).map(_.toFloat)
    Ndarray(Tensor(array, Array(array.length)))
  }
}
