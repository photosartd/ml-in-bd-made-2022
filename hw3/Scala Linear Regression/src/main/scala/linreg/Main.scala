package linreg

import breeze.linalg._
import breeze.numerics.abs
import breeze.stats.mean

import java.io._

object Main extends App {
  def mae(y_true: DenseVector[Double], y_pred: DenseVector[Double]): Double = {
    mean(abs(y_true - y_pred))
  }

  val matrix = csvread(new File("/Users/dmitry.trofimov/Documents/MADE/winequality_red.csv"),',')
  //x, y
  val y: DenseVector[Double] = matrix(::, -1)
  val x: DenseMatrix[Double] = matrix(::, 0 to -2)

  val model = new LinearRegression
  model.fit(x, y)
  val prediction = model.predict(x)
  println(prediction)


  val maERR = mae(y, prediction)
  println(s"MAE: ${maERR}")

}
