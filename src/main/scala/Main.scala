package es.upm.airplane
import com.github.nscala_time.time.Imports._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{AnalysisException, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.feature.{Imputer, StandardScaler, VectorAssembler}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.log4j.{Level, Logger}
import scala.math.Pi

object Main extends App{
  Logger.getLogger("org").setLevel(Level.WARN)

  if (args.length == 0) {
    println("Please introduce at least one parameter with the path file")
    System.exit(0)
  }
  val filename = args(0)

  val spark = SparkSession.builder()
    .master("local[*]")
    .appName("Airplane Delay")
    .getOrCreate()

  import spark.implicits._

  val sc = spark.sparkContext

  // Create schema and read csv
  val schema = StructType(Array(
    StructField("Year",StringType,true),
    StructField("Month",StringType,true),
    StructField("DayofMonth",StringType,true),
    StructField("DayofWeek", IntegerType,true),
    StructField("DepTime",StringType,true),
    StructField("CRSDepTime",StringType,true),
    StructField("ArrTime",StringType,true),
    StructField("CRSArrTime",StringType,true),
    StructField("UniqueCarrier",StringType,true),
    StructField("FlightNum",IntegerType,true),
    StructField("TailNum",StringType,true),
    StructField("ActualElapsedTime",StringType,true),
    StructField("CRSElapsedTime",IntegerType,true),
    StructField("AirTime",StringType,true),
    StructField("ArrDelay",IntegerType,true),
    StructField("DepDelay",IntegerType,true),
    StructField("Origin",StringType,true),
    StructField("Dest",StringType,true),
    StructField("Distance",IntegerType,true),
    StructField("TaxiIn",IntegerType,true),
    StructField("TaxiOut",DoubleType,true),
    StructField("Cancelled",IntegerType,true),
    StructField("CancellationCode",StringType,true),
    StructField("Diverted",StringType,true),
    StructField("CarrierDelay",StringType,true),
    StructField("WeatherDelay",StringType,true),
    StructField("NASDelay",StringType,true),
    StructField("SecurityDelay",StringType,true),
    StructField("LateAircraftDelay",StringType,true),
  ))

  // Read the data
  var df = spark.emptyDataFrame
  try{
    df = spark.read.option("header",true).option("delimiter", ",").option("enforceSchema", false).schema(schema).csv(filename)
    df.head(1)
  }catch {
    case ex: AnalysisException =>
      println(s"Path does not exist $filename")
      System.exit(1)
    case unknown: Exception =>
      println("Schema mismatch")
      System.exit(1)
  }

  /////////////////////////////////////////////////
  ////////// INITIAL TRANSFORMATIONS //////////////
  /////////////////////////////////////////////////

  // Remove forbidden columns and not useful ones
  val dfRemoveColumns= df.drop( "ArrTime","ActualElapsedTime",
    "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay", "CancellationCode", "Year")

  //Filter out not arriving flights
  val dfFilterArrDelay = dfRemoveColumns.filter($"ArrDelay".isNotNull)

  //Filter out cancelled flights
  val dfFilterCancelled = dfFilterArrDelay.filter($"Cancelled" === 0)
    .drop("Cancelled")

  //Filter outliers (ArrDelay>80 and ArrDelay<-21)
  val dfFilterOutliers = dfFilterCancelled.filter(($"ArrDelay"<=80) && ($"ArrDelay">=(-20)))

  //Filter out null values
  val dfNullDropped = dfFilterOutliers.na.drop("any", Seq("Month","DayofMonth","DayofWeek","DepTime","CRSDepTime",
    "CRSArrTime", "UniqueCarrier","FlightNum","TailNum","CRSElapsedTime","ArrDelay","DepDelay","Origin","Dest","Distance"))

  //Merge full date and manage leap-year
  val dfDate = dfNullDropped
    .withColumn("DayofMonth",
      when($"DayofMonth"===29 && $"Month"===2, 28)
        .otherwise($"DayofMonth"))
    .withColumn("DateString",
    concat(lpad($"DayofMonth", 2, "0"),
      lit("-"),
      lpad($"Month", 2, "0"),
      lit("-"),
      lit("2002")))
    .withColumn("Date", to_date($"DateString", "dd-MM-yyyy"))
    .withColumn("DayofYear", date_format($"Date", "D"))
    .withColumn("DayofYear", ($"DayofYear" -1).cast(IntegerType))
    .drop("DayofMonth", "Year", "DateString", "Month")

  //Complete times
  val dfTimes = dfDate
    .withColumn("DepTime", lpad($"DepTime", 4, "0"))
    .withColumn("CRSDepTime", lpad($"CRSDepTime", 4, "0"))
    .withColumn("CRSArrTime", lpad($"CRSArrTime", 4, "0"))

  //Cyclical features encoding
  def cyclicalEncodingTime (dfIni: DataFrame, columnName:String) : DataFrame = {
    // Assign 0 to 00:00 until 1439 to 23:59 (= 24 hours x 60 minutes)
    // Create table with encoding
    val ini = DateTime.now().hour(0).minute(0).second(0)
    val values = (0 until 1440).map(ini.plusMinutes(_)).toList
    val hourAndMinutes = values.map(x=>(x.getHourOfDay().toString, x.getMinuteOfHour().toString))
    val columns = Seq("Hour","Min")
    val dfCyclical = hourAndMinutes.toDF(columns:_*)
      .withColumn("Time",
        concat(lpad($"Hour", 2, "0"),
          lpad($"Min", 2, "0")))
      .drop("Hour", "Min")
      .withColumn("id", monotonically_increasing_id())
      .withColumn("idNorm",
        round(lit(2)*lit(Pi)*$"id"/lit(1439),6))
      .withColumn("x" + columnName, round(cos($"idNorm"),6))
      .withColumn("y" + columnName, round(sin($"idNorm"),6))
      .drop("id", "idNorm")

    val dfOut = dfIni.join(dfCyclical, dfIni(columnName)===dfCyclical("Time"), "inner")
      .drop("Time")
    return dfOut
  }

  val dfDepTimeEncoded = cyclicalEncodingTime(dfTimes, "DepTime")
    .drop("DepTime","CRSDepTime","CRSArrTime")
  //val dfCRSDepTimeEncoded = cyclicalEncodingTime(dfDepTimeEncoded, "CRSDepTime")
  //val dfCRSArrTimeEncoded = cyclicalEncodingTime(dfCRSDepTimeEncoded, "CRSArrTime")

  def cyclicalEncodingDate(dfIni: DataFrame, columnName:String) : DataFrame = {
    val dfCyclical = (0 until 365).toList.toDF("Days")
      .withColumn("idNorm",
        round(lit(2)*lit(Pi)*$"Days"/lit(364),6))
      .withColumn("x" + columnName, round(cos($"idNorm"),6))
      .withColumn("y" + columnName, round(sin($"idNorm"),6))
      .drop("idNorm")

    val dfOut = dfIni.join(dfCyclical, dfIni(columnName)===dfCyclical("Days"), "inner")
      .drop("Days")
    return dfOut
  }

  val dfDateEncoded = cyclicalEncodingDate(dfDepTimeEncoded, "DayofYear")
    .drop("Date", "DayofYear")
    .withColumnRenamed("ArrDelay","label")

  //Encoding for categorical variables
  def meanTargetEncoding(dfIni: DataFrame, columnName:String) : DataFrame = {
    //Return a dataframe with the og. variable and the variable with mean target encoding as columnNameEncoded
    val dfOut = dfIni.withColumn(columnName + "Encoded",
      round(avg("label").over(Window.partitionBy(columnName)),6))
      .select(columnName, columnName + "Encoded")
      .distinct()
    return dfOut
  }

  // Split training, test and validation
  val Array(trainingIni, testIni, validationIni) = dfDateEncoded.randomSplit(Array(0.6,0.2,0.2),seed=1)

  // Apply encoding function to the train and join the result with training, test and val
  val dayofWeekEncoding = meanTargetEncoding(trainingIni, "DayofWeek").cache()
  dayofWeekEncoding.show(false)
  val originEncoding = meanTargetEncoding(trainingIni, "Origin").cache()
  originEncoding.show(false)
  val destEncoding = meanTargetEncoding(trainingIni, "Dest").cache()
  destEncoding.show(false)
  //val uniqueCarrierEncoding = meanTargetEncoding(trainingIni, "UniqueCarrier").cache()
  //uniqueCarrierEncoding.show(false)
  //val flightNumEncoding = meanTargetEncoding(trainingIni, "FlightNum").cache()
  //flightNumEncoding.show(false)
  //val tailNumEncoding = meanTargetEncoding(trainingIni, "TailNum").cache()
  //tailNumEncoding.show(false)

  // Join each sample with target encoding for categorical variables
  val training = trainingIni
    .join(dayofWeekEncoding, Seq("DayofWeek"), "left")
    .join(originEncoding, Seq("Origin"), "left")
    .join(destEncoding, Seq("Dest"), "left")
    .drop("UniqueCarrier", "DayofWeek", "FlightNum", "TailNum", "Origin", "Dest").cache()
  training.show(false)

  val test = testIni
    .join(dayofWeekEncoding, Seq("DayofWeek"), "left")
    .join(originEncoding, Seq("Origin"), "left")
    .join(destEncoding, Seq("Dest"), "left")
    .drop("UniqueCarrier", "DayofWeek", "FlightNum", "TailNum", "Origin", "Dest").cache()
  test.show(false)

  val validation = validationIni
    .join(dayofWeekEncoding, Seq("DayofWeek"), "left")
    .join(originEncoding, Seq("Origin"), "left")
    .join(destEncoding, Seq("Dest"), "left")
    .drop("UniqueCarrier", "DayofWeek", "FlightNum", "TailNum", "Origin", "Dest").cache()
  validation.cache()

  /////////////////////////////////////////////////
  ////////// MODEL PIPELINE ///////////////////////
  /////////////////////////////////////////////////

  //Fill null values
  val colsNames = Array("TaxiOut",  "DayofWeekEncoded", "OriginEncoded", "DestEncoded")
  val imputer = new Imputer()
    .setInputCols(colsNames)
    .setOutputCols(colsNames.map(c => s"${c}Imputed"))
    .setStrategy("median")

  //Feautures Assembler
  val assembler = new VectorAssembler()
    .setInputCols(Array("CRSElapsedTime","DepDelay", "Distance", "TaxiOutImputed", "xDayofYear", "yDayofYear",
      "xDepTime", "yDepTime","DayofWeekEncodedImputed", "OriginEncodedImputed", "DestEncodedImputed"))
    .setOutputCol("features")

  //Apply a normalizer
  val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)

  //Linear Regression
  val lr = new LinearRegression()
    .setFeaturesCol("scaledFeatures")
    .setPredictionCol("predictionLr")
    .setLabelCol("label")
    .setElasticNetParam(0)
    .setStandardization(false)

  //Random Forest
  val rf = new RandomForestRegressor()
    .setFeaturesCol("scaledFeatures")
    .setLabelCol("label")
    .setPredictionCol("predictionRf")
    .setMaxDepth(12)
    .setNumTrees(100)
    .setMaxBins(128)

  // Pipeline
  val pipeline = new Pipeline()
    .setStages(Array(imputer, assembler, scaler, lr, rf))

  // Fit the model
  val model = pipeline.fit(training)

  // Evaluate the model
  val trainPredictions = model.transform(training)
  val testPredictions = model.transform(test)
  val validationPredictions = model.transform(validation)

  // Print the metrics
  def getMetrics(sample: DataFrame, sampleName: String, model:String):(String, String, Double, Double)={
    //Return a tuple with the model, sample, rmse and r2
    val evaluatorRmse = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction" + model)
      .setMetricName("rmse")
    val evaluatorR2 = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction" + model)
      .setMetricName("r2")

    val rmse = evaluatorRmse.evaluate(sample)
    val r2 = evaluatorR2.evaluate(sample)

    return (model, sampleName, rmse, r2)
  }

  /////////////////////////////////////////////////
  ////////// RESULTS //////////////////////////////
  /////////////////////////////////////////////////

  val results  = spark.createDataFrame(Seq(
    getMetrics(trainPredictions, "Training", "Lr"),
    getMetrics(testPredictions, "Test", "Lr"),
    getMetrics(validationPredictions, "Validation", "Lr"),
    getMetrics(trainPredictions, "Training", "Rf"),
    getMetrics(testPredictions, "Test", "Rf"),
    getMetrics(validationPredictions, "Validation", "Rf")
  )).toDF("Model","Sample","RMSE","R-squared")
    .withColumn("RMSE", round($"RMSE",3))
    .withColumn("R-squared", round($"R-squared",3))

  println(results.show)

}

