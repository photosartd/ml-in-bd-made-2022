package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

class LinearRegressionTest extends AnyFlatSpec with should.Matchers {

  val delta: Double = 0.001

  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val trainData: DataFrame = LinearRegressionTest._trainData
  lazy val w: DenseVector[Double] = LinearRegressionTest._w

  "Model" should "linreg input data" in {
    val model = new LinearRegressionModel(
      weights = Vectors.fromBreeze(breezeVector = w).toDense,
      bias = 0.0
    ).setInputCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")

    val vectors: Array[Vector] = model.transform(data).select("predictions").collect().map(_.getAs[Vector](0))

    vectors.length should be(2)
    vectors(0)(0) should be((1.5 * 0.5) + (0.3 * -0.3) + (-0.7 * 0.7) +- delta)
    vectors(1)(0) should be((1.5 * -0.9) + (0.3 * 0.0) + (-0.7 * 0.1) +- delta)

  }

  "Estimator" should "calculate weights" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")
      .setMaxIter(1000)
      .setStepSize(1.0)

    val model = estimator.fit(trainData)
    val w_calc = model.weights.asBreeze

    w_calc.size should be(3)
    w_calc(0) should be(w(0) +- delta)
    w_calc(1) should be(w(1) +- delta)
    w_calc(2) should be(w(2) +- delta)

  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinearRegression()
          .setInputCol("features")
          .setLabelCol("label")
          .setOutputCol("predictions")
          .setMaxIter(1000)
          .setStepSize(1.0)
      )
    )
    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(trainData).stages(0).asInstanceOf[LinearRegressionModel]

    val w_calc = model.weights.asBreeze

    w_calc.size should be(3)
    w_calc(0) should be(w(0) +- delta)
    w_calc(1) should be(w(1) +- delta)
    w_calc(2) should be(w(2) +- delta)

  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinearRegression()
          .setInputCol("features")
          .setLabelCol("label")
          .setOutputCol("predictions")
          .setMaxIter(1000)
          .setStepSize(1.0)
      )
    )

    val model = pipeline.fit(trainData)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    val vectors: Array[Vector] = model.transform(data).select("predictions").collect().map(_.getAs[Vector](0))

    vectors.length should be(2)
    vectors(0)(0) should be((1.5 * 0.5) + (0.3 * -0.3) + (-0.7 * 0.7) +- delta)
    vectors(1)(0) should be((1.5 * -0.9) + (0.3 * 0.0) + (-0.7 * 0.1) +- delta)

  }

}

object LinearRegressionTest extends WithSpark {

  lazy val _x: DenseMatrix[Double] = DenseMatrix.rand(100000, 3)
  lazy val _w: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
  lazy val _y: DenseVector[Double] = _x * _w

  lazy val _trainData: DataFrame = {
    import sqlc.implicits._
    lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(_x, _y.asDenseMatrix.t)
    lazy val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "label")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")

    lazy val res: DataFrame = assembler
      .transform(df)
      .select("features", "label")
    res
  }

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    Seq(
      Tuple1(Vectors.dense(0.5, -0.3, 0.7)),
      Tuple1(Vectors.dense(-0.9, 0.0, 0.1))
    ).toDF("features")
  }
}
