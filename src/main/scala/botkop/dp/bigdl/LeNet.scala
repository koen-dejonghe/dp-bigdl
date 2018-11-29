package botkop.dp.bigdl

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

object LeNet extends App {

  val numSamples = 16
  val numClasses = 10
  val lr = 1e-1f

  val input = Tensor[Float](numSamples, 28 * 28).randn()
  val target = Tensor[Float](
    Array.fill(numSamples)(Random.nextInt(numClasses).toFloat + 1),
    Array(numSamples))

  val model = Sequential[Float]()
  model
    .add(Reshape(Array(1, 28, 28)))
    .add(SpatialConvolution(1, 6, 5, 5))
    .add(SpatialMaxPooling(2, 2, 2, 2))
    .add(ReLU())
    .add(SpatialConvolution(6, 12, 5, 5))
    .add(SpatialMaxPooling(2, 2, 2, 2))
    .add(Reshape(Array(12 * 4 * 4)))
    .add(Linear(12 * 4 * 4, 100))
    .add(ReLU())
    .add(Linear(100, numClasses))
    .add(LogSoftMax())

  val criterion = ClassNLLCriterion[Float]()
  val optimizer = new Adam[Float](learningRate = 1e-2)


  def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
    val r0 = model.forward(x)
    val loss = criterion.forward(r0, target)
    println(loss)
    val dl = criterion.backward(r0, target)
    val dr0 = model.backward(input, dl)
    (loss, dr0.toTensor)
  }


  for (_ <- 1 to 1000) {

    /*
    val r0 = model.forward(input)
    val loss = criterion.forward(r0, target)
    println(loss)
    val dl = criterion.backward(r0, target)
    val dr0 = model.backward(input, dl)

    val ps = model.parameters()._1
    val dps = model.parameters()._2

    ps.zip(dps).foreach { case (p: Tensor[Float], dp: Tensor[Float]) =>
      p.sub(lr, dp)
    }
    model.zeroGradParameters()
    */


    model.modules.foreach { f: AbstractModule[Activity, Activity, Float] =>
      optimizer.optimize(f)
    }

    optimizer.optimize(feval, input)

//    val ps = model.parameters()._1
//    val dps = model.parameters()._2
//    ps.zip(dps).foreach { case (p: Tensor[Float], dp: Tensor[Float]) =>
//      p.sub(dp)
//    }

  }

  val r0 = model.evaluate().forward(input)
  println(r0)
  println(target)


}
