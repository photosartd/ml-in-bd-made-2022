package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector => BDenseVector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasMaxIter, HasOutputCol, HasStepSize}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.Row.empty.schema
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit

trait LinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol with HasMaxIter with HasStepSize {
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  setDefault(maxIter -> 10000, stepSize -> 0.1)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LogReg"))

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    //1. Add encoder for implicit casting
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    //2. Add bias column
    val datasetNew: Dataset[_] = dataset.withColumn("bias", lit(1))

    //3. Transform to vector
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "bias", $(labelCol)))
      .setOutputCol("inputVector")

    val inputVectors: Dataset[Vector] = assembler
      .transform(datasetNew)
      .select("inputVector")
      .as[Vector]

    //4. Get num features and construct weights vector
    val numFeats: Int = MetadataUtils.getNumFeatures(dataset, $(inputCol))
    var weights: BDenseVector[Double] = BDenseVector.rand[Double](numFeats + 1)

    //5. Start learning loop
    for (_ <- 0 until $(maxIter)) {
      //6. mapPartitions to create MultivariateOnlineSummarizer inside for counting means for weights
      val gradient: MultivariateOnlineSummarizer = inputVectors.rdd.mapPartitions(
        (data: Iterator[Vector]) => {
          //7. Create MultivariateOnlineSummarizer ONLY here, not in map()
          val summarizer: MultivariateOnlineSummarizer = new MultivariateOnlineSummarizer()
          //8. Iterate over Vectors and calc grad for each
          data.foreach(v => {
            val X: BDenseVector[Double] = v.asBreeze(0 until weights.size).toDenseVector
            val y: Double = v.asBreeze(weights.size)
            val grad: BDenseVector[Double] = X * (sum(X * weights) - y)
            summarizer.add(fromBreeze(grad))
          })
          Iterator(summarizer)
        }
      ).reduce(_ merge _)

      //9. Update weights
      weights = weights - $(stepSize) * gradient.mean.asBreeze

    }
    copyValues(
      new LinearRegressionModel(
        Vectors.fromBreeze(weights(0 until weights.size - 1)).toDense,
        weights(weights.size - 1)
      )
    ).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel]  = defaultCopy(extra = extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights: DenseVector,
                                           val bias: Double
                                         ) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("LogRegModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val solveUDF = dataset.sqlContext.udf.register(
      uid + "_solve",
      (x: Vector) => {
        Vectors.fromBreeze(BDenseVector(weights.asBreeze.dot(x.asBreeze)) + bias)
      }
    )
    dataset.withColumn($(outputCol), solveUDF(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) = weights.asInstanceOf[Vector] -> Vectors.fromBreeze(BDenseVector(bias))
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val meta = DefaultParamsReader.loadMetadata(path, sc = sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val weights = vectors.select(vectors("_1").as[Vector]).first()
      val bias = vectors.select(vectors("_2").as[Vector]).first()(0)

      val model = new LinearRegressionModel(weights = weights.toDense, bias = bias)
      meta.getAndSetParams(model)
      model
    }
  }
}
