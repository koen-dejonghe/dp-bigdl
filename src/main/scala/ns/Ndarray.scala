package ns

import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Tensor}

case class Ndarray(data: Tensor[Float]) {

  def shape: List[Int] = data.size().toList
  def reshape(newShape: List[Int]): Ndarray =
    Ndarray(data.resize(newShape.toArray))
  def reshape(newShape: Int*): Ndarray = reshape(newShape.toList)
  def shapeLike(other: Ndarray): Ndarray = reshape(other.shape)

  def copy(): Ndarray = Ndarray(data.clone())

  def :=(other: Ndarray): Unit = data.copy(other.data)
  def +(other: Ndarray): Ndarray = Ndarray(data + other.data)
  def +=(other: Ndarray): Unit = data.add(other.data)
  def -(other: Ndarray): Ndarray = Ndarray(data - other.data)
  def -=(other: Ndarray): Unit = data.add(-other.data)
  def *(other: Ndarray): Ndarray = Ndarray(data * other.data)
  def *=(other: Ndarray): Unit = data.cmul(other.data)
  def /(other: Ndarray): Ndarray = Ndarray(data / other.data)
  def /=(other: Ndarray): Unit = data.cdiv(other.data)
  def dot(other: Ndarray): Ndarray =
    Ndarray(DenseTensorMath.mul(data, other.data))

  def sameShape(other: Ndarray): Boolean = shape == other.shape
  def sameData(other: Ndarray): Boolean =
    data.storage().array() sameElements other.data.storage().array()


}
