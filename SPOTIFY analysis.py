from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, LinearSVC, MultilayerPerceptronClassifier
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
        self.cv_model = None
        self.coefficients = None

    def fit(self, model, label_col):
        param_grid = self.param_grid(model)
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol=label_col, metricName='areaUnderROC')
        cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=123)
        self.cv_model = cv.fit(self.data_processor.data)
        return self.cv_model.bestModel

    def transform(self, model, predictions):
        return model.transform(predictions)

    def param_grid(self, model):
        raise NotImplementedError("Subclasses should implement param_grid method.")

    def best_params(self, model_type):
        raise NotImplementedError("Subclasses should implement best_params method.")

    def coeff(self, label_col):
        raise NotImplementedError("Subclasses should implement best_params method.")


class LogisticRegressionTrainer(ModelTrainer):
    def __init__(self, data_processor):
        super().__init__(data_processor)
        self.cv_model = None

    def param_grid(self, model):
        return ParamGridBuilder() \
            .addGrid(model.regParam, [1.0, 0.1, 10.0]) \
            .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()

    def coeff(self, label_col):
        self.coefficients = self.cv_model.bestModel.coefficients.toArray()
        coefficients = self.coefficients
        column_names = [col for col in self.data_processor.data.columns if
                        col.endswith("_index") and col != f"{label_col}_index"]
        coefficient_dict = {column_names[i]: coefficients[i] for i in range(len(column_names))}
        sorted_coefficients = sorted(coefficient_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_coefficients

    def best_params(self, model_type):
        best_params = {}
        best_params['regParam'] = self.cv_model.bestModel._java_obj.getRegParam()
        best_params['elasticNetParam'] = self.cv_model.bestModel._java_obj.getElasticNetParam()
        return best_params


class LinearSVCTrainer(ModelTrainer):
    def __init__(self, data_processor):
        super().__init__(data_processor)
        self.cv_model = None

    def param_grid(self, model):
        return ParamGridBuilder() \
            .addGrid(model.regParam, [1.0, 0.1, 10.0]) \
            .build()

    def coeff(self, label_col):
        self.coefficients = self.cv_model.bestModel.coefficients.toArray()
        coefficients = self.coefficients
        column_names = [col for col in self.data_processor.data.columns if
                        col.endswith("_index") and col != f"{label_col}_index"]
        coefficient_dict = {column_names[i]: coefficients[i] for i in range(len(column_names))}
        sorted_coefficients = sorted(coefficient_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_coefficients

    def best_params(self, model_type):
        best_params = {}
        best_params['regParam'] = self.cv_model.bestModel._java_obj.getRegParam()
        return best_params


class MultilayerPerceptronTrainer(ModelTrainer):
    def __init__(self, data_processor):
        super().__init__(data_processor)
        self.cv_model = None

    def param_grid(self, model):
        return ParamGridBuilder() \
            .addGrid(model.layers, [[len(self.data_processor.data.select('features').first()[0]), 64, 32, 2]]) \
            .build()

    def coeff(self, label_col):
        return None

    def best_params(self, model_type):
        return None



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

    # LOGISTIC REGRESSION
    start_time = time.perf_counter()
    lr_trainer = LogisticRegressionTrainer(data_processor)
    print("Logistic Regression Model:")
    lr_model = lr_trainer.fit(LogisticRegression(featuresCol='features', labelCol='Top50_dummy'), 'Top50_dummy')
    lr_predictions = lr_trainer.transform(lr_model, data_processor.data)
    model_evaluator = ModelEvaluator(lr_predictions)
    accuracy, f1_score, precision, recall = model_evaluator.evaluate_model('Top50_dummy')
    print(f"Accuracy: {accuracy}, \nF1 Score: {f1_score}, \nWeighted Precision: {precision}, \nWeighted Recall: {recall}")
    lr_best_params = lr_trainer.best_params('Logistic Regression')
    lr_coefficients = lr_trainer.coeff('Top50_dummy')
    print("Best parameters: ", lr_best_params)
    print("Coefficients:", lr_coefficients)
    model_evaluator.plot_confusmx('Top50_dummy', 'Logistic Regression')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elaboration time Logistic Regression: {minutes} minutes and {seconds} seconds")
    print()

    # SVM
    start_time = time.perf_counter()
    svm_trainer = LinearSVCTrainer(data_processor)
    print("SVM Model:")
    svm_model = svm_trainer.fit(LinearSVC(featuresCol='features', labelCol='Top50_dummy'), 'Top50_dummy')
    svm_predictions = svm_trainer.transform(svm_model, data_processor.data)
    model_evaluator = ModelEvaluator(svm_predictions)
    accuracy, f1_score, precision, recall = model_evaluator.evaluate_model('Top50_dummy')
    print(f"Accuracy: {accuracy}, \nF1 Score: {f1_score}, \nWeighted Precision: {precision}, \nWeighted Recall: {recall}")
    svm_best_params = svm_trainer.best_params('SVM')
    svm_coefficients = svm_trainer.coeff('Top50_dummy')
    print("Best parameters: ", svm_best_params)
    print("Coefficients:", svm_coefficients)
    model_evaluator.plot_confusmx('Top50_dummy', 'SVM')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elaboration time SVM: {minutes} minutes and {seconds} seconds")
    print()

    # NEURAL NETWORK
    start_time = time.perf_counter()
    nn_trainer = MultilayerPerceptronTrainer(data_processor)
    print("Neural Network Model:")
    nn_model = nn_trainer.fit(MultilayerPerceptronClassifier(featuresCol='features', labelCol='Top50_dummy'),'Top50_dummy')
    nn_predictions = nn_trainer.transform(nn_model, data_processor.data)
    model_evaluator = ModelEvaluator(nn_predictions)
    accuracy, f1_score, precision, recall = model_evaluator.evaluate_model('Top50_dummy')
    print(f"Accuracy: {accuracy}, \nF1 Score: {f1_score}, \nWeighted Precision: {precision}, \nWeighted Recall: {recall}")
    model_evaluator.plot_confusmx('Top50_dummy', 'Neural Network')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elaboration time Neural Network: {minutes} minutes and {seconds} seconds")

    spark.stop()





#RESULTS
'''
Logistic Regression Model:
Accuracy: 0.7060533425539023, 
F1 Score: 0.586141320246044, 
Weighted Precision: 0.7129765678260117, 
Weighted Recall: 0.7060533425539023
Best parameters:  {'regParam': 0.1, 'elasticNetParam': 0.0}
Coefficients: [('Album/Single_index', 0.11162132891553857), ('Explicit_index', 0.06542688779800432), ('Cluster_index', -0.05247541120001101),
 ('mode_index', 0.047345553574151725), ('Celebrate_index', 0.030628994967937553), ('LDA_Topic_index', 0.018135803551197968), 
 ('Genre_new_index', -0.017659935609363544), ('Track_number_index', -0.012396713467940199), ('time_signature_index', -0.011488209056342444), 
 ('Tracks_in_album_index', -0.006995225606904615), ('key_index', -0.0020267047110636804), ('Country0_index', -0.0011097231314476685),
  ('Genre_index', -0.0004651175073242029), ('danceability_index', -0.00027453972221934327), ('energy_index', -0.00023903794172334573), 
  ('liveliness_index', -0.00018515681040910767), ('acoustics_index', -0.0001738796065828109), ('valence_index', -0.00014550568318329757),
   ('Release_date_index', -0.00013170580503140574), ('Days_since_release_index', -8.016789978708709e-05), ('Artist_followers_index', -7.385870200517468e-05), 
   ('instrumentalness_index', -6.772605476204929e-05), ('loudness_index', -3.820041633137325e-05), ('speechiness_index', -2.493615292290213e-05), 
   ('Artist_index', -8.981537349933107e-06), ('tempo_index', -8.771930437298352e-06), ('duration_ms_index', -3.5085938775580984e-06), 
   ('Album9_index', -3.3047628285085835e-06), ('Title_index', -2.6913141828733045e-06), ('Uri_index', -3.50117240161987e-07)]
Elaboration time Logistic Regression: 9 minutes and 24 seconds

SVM Model:
Accuracy: 0.7052270076714352, 
F1 Score: 0.5833183853079524, 
Weighted Precision: 0.4973451323492065, 
Weighted Recall: 0.7052270076714352
Best parameters:  {'regParam': 1.0}
Coefficients: [('Cluster_index', -7.139975848423461e-08),
'Celebrate_index', -4.3622278599149676e-08), ('Album/Single_index', 3.15170406046276e-08), 
('LDA_Topic_index', 3.087759872669871e-08), ('Track_number_index', -1.3252686219505844e-08), ('Genre_new_index', -8.698721681108367e-09), 
('Explicit_index', -5.418134380876136e-09), ('Country0_index', 5.035702278410529e-09), ('Days_since_release_index', -2.101715032655418e-09),
 ('Release_date_index', 1.9886596533959027e-09), ('Tracks_in_album_index', -1.5463790499678259e-09), ('Genre_index', 7.001738528187337e-10), 
 ('Artist_followers_index', -1.8664775080904706e-10), ('danceability_index', -6.637790617164358e-11), 
 ('liveliness_index', -6.041052347022279e-11), ('energy_index', 5.0077031777141654e-11), ('valence_index', -2.537360271269204e-11), 
 ('acoustics_index', -2.2288770046802473e-11), ('loudness_index', -1.1999522293455141e-11), 
 ('instrumentalness_index', -9.390283229334597e-12), ('Artist_index', 7.965591391034898e-12), ('tempo_index', -7.427766141647492e-12), 
 ('Title_index', 3.795022526565822e-12), ('duration_ms_index', 1.908121105713283e-12), ('Uri_index', 1.2132659482595514e-12), 
 ('Album9_index', -2.4582546810972145e-13), ('key_index', 0.0), ('mode_index', -0.0), ('speechiness_index', 0.0), ('time_signature_index', -0.0)]
Elaboration time SVM: 48 minutes and 32 seconds

Neural Network Model:
Accuracy: 0.7358248404470413, 
F1 Score: 0.6840691706654078, 
Weighted Precision: 0.7196331079880173, 
Weighted Recall: 0.7358248404470413
Elaboration time Neural Network: 19 minutes and 3 seconds

'''