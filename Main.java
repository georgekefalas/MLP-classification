import java.io.*;
import java.util.*;

public class Main {

    public static void loadFile(String filename, List<double[]> X_out, List<Integer> labels_out) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] parts = line.split(",");
            double x1 = Double.parseDouble(parts[0]);
            double x2 = Double.parseDouble(parts[1]);
            int lab = Integer.parseInt(parts[2]);
            X_out.add(new double[]{x1,x2});
            labels_out.add(lab - 1);
        }
        br.close();
    }

    public static void exportTestClassificationResults(
        String outFilename,
        MLP_SDT mlp,
        List<double[]> Xtest,
        int[] labelsTest) throws IOException {

        BufferedWriter bw = new BufferedWriter(new FileWriter(outFilename));

        for (int i = 0; i < Xtest.size(); i++) {
            double[] x = Xtest.get(i);
            int trueLabel = labelsTest[i];
            int predictedLabel = mlp.predict(x);

            char mark = (predictedLabel == trueLabel) ? '+' : '-';

            bw.write(x[0] + "," + x[1] + "," + mark);
            bw.newLine();
        }

        bw.close();
        System.out.println("Test classification results written to: " + outFilename);
    }


    public static void main(String[] args) throws IOException {
        String trainFile = "train.txt";
        String testFile  = "test.txt";

        List<double[]> Xtrain = new ArrayList<>();
        List<Integer> Ytrain = new ArrayList<>();
        List<double[]> Xtest  = new ArrayList<>();
        List<Integer> Ytest  = new ArrayList<>();

        System.out.println("Loading datasets...");
        loadFile(trainFile, Xtrain, Ytrain);
        loadFile(testFile, Xtest, Ytest);
        System.out.printf("Loaded: train=%d, test=%d%n", Xtrain.size(), Xtest.size());

        int[] labelsTrain = Ytrain.stream().mapToInt(Integer::intValue).toArray();
        int[] labelsTest  = Ytest.stream().mapToInt(Integer::intValue).toArray();

        MLP_SDT mlp = new MLP_SDT(1926);
        mlp.learningRate = 0.02;
        mlp.maxEpochs = 1000;
        mlp.minEpochs = 800;
        mlp.stopDiffThreshold = 0.0001;

        System.out.println("Starting training...");
        mlp.train(Xtrain, labelsTrain);

        System.out.println("Evaluating on test set...");
        double acc = mlp.evaluateAccuracy(Xtest, labelsTest);
        System.out.printf("Test accuracy: %.4f (%.2f%%)%n", acc, acc*100.0);

        //exportTestClassificationResults("test_results_best_model.txt", mlp, Xtest, labelsTest);

    }
}
