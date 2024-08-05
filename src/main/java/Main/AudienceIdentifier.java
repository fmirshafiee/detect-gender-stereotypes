package Main;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class AudienceIdentifier {

    private static final String MODEL_PATH = "model.zip";
    private static final String SPARK_MASTER = "local";

    public static void main(String[] args) throws Exception {

        String inputText = "Watching wrestling matches is my favorite pastime";

        // Initialize Spark context
        SparkConf conf = new SparkConf().setAppName("TFIDFExample").setMaster(SPARK_MASTER);
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Convert input text to RDD
        JavaRDD<String> textRDD = sc.parallelize(Arrays.asList(inputText));

        // Tokenize input text
        JavaRDD<List<String>> wordsRDD = textRDD.map(s -> Arrays.asList(s.split(" ")));

        // Compute TF-IDF features
        HashingTF hashingTF = new HashingTF();
        JavaRDD<Vector> tf = hashingTF.transform(wordsRDD);
        tf.cache();
        IDF idf = new IDF();
        org.apache.spark.mllib.feature.IDFModel idfModel = idf.fit(tf);
        JavaRDD<Vector> tfidf = idfModel.transform(tf);

        // Get the TF-IDF feature array
        Vector tfidfVector = tfidf.first();
        double[] features = tfidfVector.toArray();

        // Load the trained model
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(MODEL_PATH);

        // Convert features to INDArray
        INDArray input = Nd4j.create(new double[][]{features});

        // Predict the audience
        INDArray output = model.output(input);
        int predictedClass = Nd4j.argMax(output, 1).getInt(0);

        String[] classes = {"female", "male", "neutral"};
        String predictedLabel = classes[predictedClass];

        System.out.println("man" + predictedLabel);

        sc.close();
    }
}
