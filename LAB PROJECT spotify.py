from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


class SpotifyDataProcessor:
    def __init__(self, spark):
        self.spark = spark
        self.data = None

    def load_data(self, path):
        self.data = self.spark.read.csv(path, header=True, inferSchema=True)

    def preprocess_data(self):
        indexers = []
        for col_name, col_type in self.data.dtypes:
            if col_type == "string":
                indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
                self.data = indexer.fit(self.data).transform(self.data)
                indexers.append(indexer)

        input_cols = [f"{col_name}_index" for col_name, col_type in self.data.dtypes if col_type == "string"]
        assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
        self.data = assembler.transform(self.data)
        return self.data


class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def model_coef(self, model, label_col):
        param_grid = self.gparam_grid(model)

        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol=label_col, metricName='areaUnderROC')
        cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=123)
        cv_model = cv.fit(self.data_processor.data)
        predictions = cv_model.transform(self.data_processor.data)

        best_params = self.gbest_params(cv_model.bestModel, type(model))

        coefficients = None
        if isinstance(model, LogisticRegression) or isinstance(model, LinearSVC):
            coefficients = cv_model.bestModel.coefficients.toArray()
            column_names = [col for col in self.data_processor.data.columns if
                            col.endswith("_index") and col != f"{label_col}_index"]
            coefficient_dict = {column_names[i]: coefficients[i] for i in range(len(column_names))}
            sorted_coefficients = sorted(coefficient_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            coefficients = sorted_coefficients
        return predictions, best_params, coefficients

    def gparam_grid(self, model):
        param_grid = None
        if isinstance(model, LogisticRegression):
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam, [1.0, 0.1, 10.0]) \
                .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
                .build()
        elif isinstance(model, LinearSVC):
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam, [1.0, 0.1, 10.0]) \
                .build()
        elif isinstance(model, MultilayerPerceptronClassifier):
            param_grid = ParamGridBuilder() \
                .addGrid(model.layers, [[len(self.data_processor.data.select('features').first()[0]), 64, 32, 2]]) \
                .build()
        else:
            return param_grid
        return param_grid

    def gbest_params(self, best_model, model_type):
        best_params = {}
        if model_type == LogisticRegression:
            best_params['regParam'] = best_model._java_obj.getRegParam()
            best_params['elasticNetParam'] = best_model._java_obj.getElasticNetParam()
        elif model_type == LinearSVC:
            best_params['regParam'] = best_model._java_obj.getRegParam()
        return best_params


class ModelEvaluator:
    def __init__(self, predictions):
        self.predictions = predictions

    def evaluate_model(self, label_col):
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol=label_col)
        accuracy = evaluator.evaluate(self.predictions, {evaluator.metricName: "accuracy"})
        f1_score = evaluator.evaluate(self.predictions, {evaluator.metricName: "f1"})
        precision = evaluator.evaluate(self.predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(self.predictions, {evaluator.metricName: "weightedRecall"})
        return accuracy, f1_score, precision, recall

    def plot_confusmx(self, label_col, model_name):
        cm = confusion_matrix(self.predictions.select("prediction").rdd.map(lambda x: x[0]).collect(),
                              self.predictions.select(label_col).rdd.map(lambda x: x[0]).collect(),
                              labels=[0.0, 1.0])
        cmplot = ConfusionMatrixDisplay(cm, display_labels=[0.0, 1.0])
        cmplot.plot(values_format='d')
        plt.title(f"Confusion Matrix {model_name}")
        plt.show()



if __name__ == "__main__":

    spark = SparkSession.builder \
        .appName("Spotify Data Processor") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    data_processor = SpotifyDataProcessor(spark)
    data_processor.load_data("DB_spotify.csv")
    data_processor.preprocess_data()

    start_time = time.perf_counter()
    model_trainer = ModelTrainer(data_processor)
    lr_model_predictions, lr_best_params, lr_coefficients = model_trainer.model_coef(
        LogisticRegression(featuresCol='features', labelCol='Top50_dummy'), 'Top50_dummy')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elaboration time Logistic Regression: {minutes} minutes and {seconds} seconds")

    start_time = time.perf_counter()
    svm_model_predictions, svm_best_params, svm_coefficients = model_trainer.model_coef(
        LinearSVC(featuresCol='features', labelCol='Explicit_true'), 'Explicit_true')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elaboration time SVM: {minutes} minutes and {seconds} seconds")

    start_time = time.perf_counter()
    nn_model_predictions, _, _ = model_trainer.model_coef(
        MultilayerPerceptronClassifier(featuresCol='features', labelCol='Top50_dummy'), 'Top50_dummy')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elaboration time Neural Network: {minutes} minutes and {seconds} seconds")
    print()

    models = {
        "Logistic Regression": (lr_model_predictions, lr_best_params, lr_coefficients, 'Top50_dummy'),
        "SVM": (svm_model_predictions, svm_best_params, svm_coefficients, 'Explicit_true'),
        "Neural Network": (nn_model_predictions, None, None, 'Top50_dummy')
    }

    for model_name, (model_predictions, best_params, coefficients, label_col) in models.items():
        print(f"{model_name} Model Evaluation:")
        model_evaluator = ModelEvaluator(model_predictions)

        accuracy, f1_score, precision, recall = model_evaluator.evaluate_model(label_col)
        print(f"Accuracy: {accuracy}, \nF1 Score: {f1_score}, \nWeighted Precision: {precision}, \nWeighted Recall: {recall}")

        print("Best parameters:")
        if best_params is not None:
                print(best_params)
        else:
            print("Parameters not available for this model.")

        print("Coefficients:")
        if coefficients is not None:
            for name, coeff in coefficients:
                print(f"{name}: {coeff}")
        else:
            print("Coefficients not available for this model.")

        model_evaluator.plot_confusmx(label_col, model_name)
        print()

    spark.stop()





# RESULTS =====================================================================================
'''
Elaboration time Logistic Regression: 8 minutes and 7 seconds
Elaboration time SVM: 40 minutes and 51 seconds
Elaboration time Neural Network: 17 minutes and 3 seconds

Logistic Regression Model Evaluation:
Accuracy: 0.7060533425539023
F1 Score: 0.586141320246044
Weighted Precision: 0.7129765678260117
Weighted Recall: 0.7060533425539023
Best parameters:
{regParam:0.1, elasticNetParam: 0.0}
Coefficients:
Album/Single_index: 0.11162132891553857
Explicit_index: 0.06542688779800432
Cluster_index: -0.05247541120001101
mode_index: 0.047345553574151725
Celebrate_index: 0.030628994967937553
LDA_Topic_index: 0.018135803551197968
Genre_new_index: -0.017659935609363544
Track_number_index: -0.012396713467940199
time_signature_index: -0.011488209056342444
Tracks_in_album_index: -0.006995225606904615

SVM Model Evaluation:
Accuracy: 0.9997479971635029
F1 Score: 0.9997480134085053
Weighted Precision: 0.9997481119724063
Weighted Recall: 0.9997479971635029
Best parameters:
{regParam: 1.0}
Coefficients:
Explicit_index: 1.532861595423319
Album/Single_index: -0.007310094148246977
mode_index: 0.006249578560206523
Celebrate_index: -0.0038912176237244074
LDA_Topic_index: 0.0020471902018289905
Genre_new_index: -0.001229910517111018
time_signature_index: 0.0011970418599278867
Track_number_index: 0.0011175865379366907
Country0_index: -0.0007275157292894498
Cluster_index: -0.00039071383002767154

Neural Network Model Evaluation:
Accuracy: 0.7383155661566051
F1 Score: 0.6886941704989773
Weighted Precision:	0.723221678582239
Weighted Recall: 0.7383155661566052
Best parameters:
Parameters not available for this model.
Coefficients:
Coefficients not available for this model.
'''