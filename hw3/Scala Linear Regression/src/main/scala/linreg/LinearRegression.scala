package linreg

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}
import breeze.stats.{mean, variance}

class LinearRegression{
  private var w: DenseVector[Double] = DenseVector.ones[Double](1)
  private var x_mean: DenseVector[Double] = DenseVector.ones[Double](1)
  private var x_var: DenseVector[Double] = DenseVector.ones[Double](1)

  def fit(x: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
    x_mean = mean(x(::, *)).t
    x_var = variance(x(::, *)).t

    val diff: DenseMatrix[Double] = x(*, ::) - x_mean
    val normalizedXWithoutIntercept: DenseMatrix[Double] = diff(*, ::) /:/ x_var
    val newCol: DenseVector[Double] = DenseVector.ones[Double](normalizedXWithoutIntercept.rows)
    val normalizedX: DenseMatrix[Double] = DenseMatrix.horzcat(normalizedXWithoutIntercept,
      new DenseMatrix[Double](normalizedXWithoutIntercept.rows, 1, newCol.toArray))

    w = (inv(normalizedX.t * normalizedX) * normalizedX.t) * y
  }

  def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    val diff: DenseMatrix[Double] = x(*, ::) - x_mean
    val normalizedXWithoutIntercept: DenseMatrix[Double] = diff(*, ::) /:/ x_var
    val newCol: DenseVector[Double] = DenseVector.ones[Double](normalizedXWithoutIntercept.rows)
    val normalizedX: DenseMatrix[Double] = DenseMatrix.horzcat(normalizedXWithoutIntercept,
      new DenseMatrix[Double](normalizedXWithoutIntercept.rows, 1, newCol.toArray))

    normalizedX * w
  }

}
