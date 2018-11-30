package scorch

import com.intel.analytics.bigdl.tensor.Tensor

trait Function[@specialized(Float, Double) T] {
  def forward(): Variable[T]
  def backward(g: Tensor[T]): Unit
}

