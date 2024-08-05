package Main;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TrainAndSaveModel {

    private static final String SPARK_MASTER = "local";
    private static final String MODEL_PATH = "model.zip";

    public static void main(String[] args) throws Exception {
        // Load text and labels from CSV
        List<String> texts = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("data.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("text")) continue; // Skip header
                String[] values = line.split(",");
                texts.add(values[0].replace("\"", "")); // Remove quotes
                labels.add(values[1].replace("\"", "")); // Remove quotes
            }
        }

        // Initialize Spark context
        SparkConf conf = new SparkConf().setAppName("TFIDFExample").setMaster(SPARK_MASTER);
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Convert texts to RDD
        JavaRDD<String> textRDD = sc.parallelize(texts);

        // Tokenize texts
        JavaRDD<List<String>> wordsRDD = textRDD.map(s -> Arrays.asList(s.split(" ")));

        // Compute TF-IDF features
        HashingTF hashingTF = new HashingTF();
        JavaRDD<Vector> tf = hashingTF.transform(wordsRDD);
        tf.cache();
        IDF idf = new IDF();
        org.apache.spark.mllib.feature.IDFModel idfModel = idf.fit(tf);
        JavaRDD<Vector> tfidf = idfModel.transform(tf);

        // Get the TF-IDF feature array
        List<Vector> tfidfList = tfidf.collect();
        double[][] features = new double[tfidfList.size()][];
        for (int i = 0; i < tfidfList.size(); i++) {
            features[i] = tfidfList.get(i).toArray();
        }

        // Convert labels to one-hot encoding
        double[][] labelMatrix = new double[labels.size()][3];
        for (int i = 0; i < labels.size(); i++) {
            if (labels.get(i).equals("female")) {
                labelMatrix[i][0] = 1.0;
                labelMatrix[i][1] = 0.0;
                labelMatrix[i][2] = 0.0;
            } else if (labels.get(i).equals("male")) {
                labelMatrix[i][0] = 0.0;
                labelMatrix[i][1] = 1.0;
                labelMatrix[i][2] = 0.0;
            } else if (labels.get(i).equals("neutral")) {
                labelMatrix[i][0] = 0.0;
                labelMatrix[i][1] = 0.0;
                labelMatrix[i][2] = 1.0;
            }
        }

        INDArray input = Nd4j.create(features);
        INDArray output = Nd4j.create(labelMatrix);

        DataSet dataSet = new DataSet(input, output);
        List<DataSet> listDs = dataSet.asList();
        DataSetIterator trainIter = new ListDataSetIterator<>(listDs, 10);

        int numInputs = features[0].length;
        int numOutputs = 3;
        int numHiddenNodes = 10;

        MultiLayerConfiguration neuralConf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(neuralConf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        int numEpochs = 30;
        for (int i = 0; i < numEpochs; i++) {
            trainIter.reset();
            model.fit(trainIter);
        }

        // Save the model
        ModelSerializer.writeModel(model, new File(MODEL_PATH), true);

        sc.close();
    }
}
