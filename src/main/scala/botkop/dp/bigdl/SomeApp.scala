package botkop.dp.bigdl

object SomeApp {

  def main(args: Array[String]): Unit = {
    val a = ns.arange(10).reshape(2, 5)
    val b = ns.arange(10).reshape(5, 2)
    val c = a dot b
    println(c)

    a.sameData(b)
  }

}
