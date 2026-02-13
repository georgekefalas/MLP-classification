import java.io.*;
import java.util.*;

/*
 Multilayer Perceptron implementation for the SDT problem (labels 0..K-1).
 3 hidden layers
 Hidden activation: "logistic", "tanh", "relu"
 Output activation: logistic
 Loss: Mean Squared Error (MSE) (Total MSE over all training samples is printed per epoch)
 Mini-batch gradient descent with batch size L
 Weight initialization uniform in (-1,1)
  Stopping rule: run at least minEpochs (800) and stop when |E_epoch - E_prev| < stopDiffThreshold
 
 Expected format for train/test files: x1,x2,label  (no header)
 */
public class MLP_SDT {

    // CONFIG (like #define)
    public static final int d = 2;             // input dimension
    public static final int K = 4;             // number of classes (SDT -> 4)
    public static final int H1 = 32;           // neurons in hidden layer 1
    public static final int H2 = 32;           // hidden layer 2
    public static final int H3 = 16;           // hidden layer 3

    // "logistic", "tanh", "relu"
    public static final String HIDDEN_ACTIVATION = "tanh";

    // Training hyperparameters (can be changed at runtime)
    public double learningRate = 0.02;
    public int batchSize = 20;        // L: mini-batch size
    public int maxEpochs = 1000;      // safety cap
    public int minEpochs = 800;       // must run at least this many epochs
    public double stopDiffThreshold = 0.001; // threshold on difference of TOTAL training error between epochs

    // Network parameters (global variables)
    private double[][] W1; private double[] b1; // H1 x d, H1
    private double[][] W2; private double[] b2; // H2 x H1, H2
    private double[][] W3; private double[] b3; // H3 x H2, H3
    private double[][] W4; private double[] b4; // K x H3, K

    // intermediate storages
    private double[] z1, a1;
    private double[] z2, a2;
    private double[] z3, a3;
    private double[] z4, a4; // output

    private Random rnd;

    // Constructor
    public MLP_SDT(long seed) {
        rnd = new Random(seed);
        initNetwork();
    }

    // Initialize weights and biases uniformly in (-1,1)
    private void initNetwork() {
        W1 = new double[H1][d]; b1 = new double[H1];
        W2 = new double[H2][H1]; b2 = new double[H2];
        W3 = new double[H3][H2]; b3 = new double[H3];
        W4 = new double[K][H3]; b4 = new double[K];

        z1 = new double[H1]; a1 = new double[H1];
        z2 = new double[H2]; a2 = new double[H2];
        z3 = new double[H3]; a3 = new double[H3];
        z4 = new double[K];  a4 = new double[K];

        for (int i = 0; i < H1; i++) {
            b1[i] = randBetween(-1,1);
            for (int j = 0; j < d; j++) W1[i][j] = randBetween(-1,1);
        }
        for (int i = 0; i < H2; i++) {
            b2[i] = randBetween(-1,1);
            for (int j = 0; j < H1; j++) W2[i][j] = randBetween(-1,1);
        }
        for (int i = 0; i < H3; i++) {
            b3[i] = randBetween(-1,1);
            for (int j = 0; j < H2; j++) W3[i][j] = randBetween(-1,1);
        }
        for (int i = 0; i < K; i++) {
            b4[i] = randBetween(-1,1);
            for (int j = 0; j < H3; j++) W4[i][j] = randBetween(-1,1);
        }
    }

    private double randBetween(double a, double b) {
        return a + (b - a) * rnd.nextDouble();
    }

    // forward-pass: compute output a4 for given x
    // This matches idea of forward(float* x, int d, float* y, int K)
    public double[] forward(double[] x) {
        // layer 1
        for (int i = 0; i < H1; i++) {
            double s = b1[i];
            for (int j = 0; j < d; j++) s += W1[i][j] * x[j];
            z1[i] = s;
            a1[i] = Activation.activate(HIDDEN_ACTIVATION, s);
        }
        // layer 2
        for (int i = 0; i < H2; i++) {
            double s = b2[i];
            for (int j = 0; j < H1; j++) s += W2[i][j] * a1[j];
            z2[i] = s;
            a2[i] = Activation.activate(HIDDEN_ACTIVATION, s);
        }
        // layer 3
        for (int i = 0; i < H3; i++) {
            double s = b3[i];
            for (int j = 0; j < H2; j++) s += W3[i][j] * a2[j];
            z3[i] = s;
            a3[i] = Activation.activate("tanh", s);          // CHANGE 3RD LAYER
        }
        // output layer (logistic activation)
        for (int i = 0; i < K; i++) {
            double s = b4[i];
            for (int j = 0; j < H3; j++) s += W4[i][j] * a3[j];
            z4[i] = s;
            a4[i] = Activation.activate("logistic", s); // logistic
        }

        // return copy
        double[] out = new double[K];
        System.arraycopy(a4, 0, out, 0, K);
        return out;
    }

    // compute gradients for a single example and accumulate into gradient arrays
    // backprop like: backprop(float *x, int d, float *t, int K)
    // but we don't update weights here; gradients are accumulated and applied after batch
    private void computeGradientsSingle(double[] x, double[] t,
                                        double[][] dW1, double[] db1,
                                        double[][] dW2, double[] db2,
                                        double[][] dW3, double[] db3,
                                        double[][] dW4, double[] db4) {

        // forward(x) must have been called before to populate a1,a2,a3,a4 and z1,z2,z3,z4
        // output delta for LINEAR output + MSE
        double[] delta4 = new double[K];
        for (int i = 0; i < K; i++) {
            delta4[i] = (a4[i] - t[i]) * Activation.derivative("logistic", z4[i], a4[i]); // MSE + logistic derivative
        }

        // gradients for W4 and b4
        for (int i = 0; i < K; i++) {
            db4[i] += delta4[i];
            for (int j = 0; j < H3; j++) dW4[i][j] += delta4[i] * a3[j];
        }

        // backprop to layer3
        double[] delta3 = new double[H3];
        for (int j = 0; j < H3; j++) {
            double s = 0.0;
            for (int i = 0; i < K; i++) s += W4[i][j] * delta4[i];
            delta3[j] = s * Activation.derivative("tanh", z3[j], a3[j]);         // CHANGE 3RD LAYER
        }
        for (int j = 0; j < H3; j++) {
            db3[j] += delta3[j];
            for (int k = 0; k < H2; k++) dW3[j][k] += delta3[j] * a2[k];
        }

        // backprop to layer2
        double[] delta2 = new double[H2];
        for (int j = 0; j < H2; j++) {
            double s = 0.0;
            for (int i = 0; i < H3; i++) s += W3[i][j] * delta3[i];
            delta2[j] = s * Activation.derivative(HIDDEN_ACTIVATION, z2[j], a2[j]);;
        }
        for (int j = 0; j < H2; j++) {
            db2[j] += delta2[j];
            for (int k = 0; k < H1; k++) dW2[j][k] += delta2[j] * a1[k];
        }

        // backprop to layer1
        double[] delta1 = new double[H1];
        for (int j = 0; j < H1; j++) {
            double s = 0.0;
            for (int i = 0; i < H2; i++) s += W2[i][j] * delta2[i];
            delta1[j] = s * Activation.derivative(HIDDEN_ACTIVATION, z1[j], a1[j]);
        }
        for (int j = 0; j < H1; j++) {
            db1[j] += delta1[j];
            for (int k = 0; k < d; k++) dW1[j][k] += delta1[j] * x[k];
        }
    }

    // apply accumulated gradients (averaged over batchCount) to update weights
    private void applyGradients(double[][] dW1, double[] db1,
                                double[][] dW2, double[] db2,
                                double[][] dW3, double[] db3,
                                double[][] dW4, double[] db4,
                                int batchCount) {
        double scale = learningRate / batchCount;
        for (int i = 0; i < H1; i++) {
            b1[i] -= scale * db1[i];
            for (int j = 0; j < d; j++) W1[i][j] -= scale * dW1[i][j];
        }
        for (int i = 0; i < H2; i++) {
            b2[i] -= scale * db2[i];
            for (int j = 0; j < H1; j++) W2[i][j] -= scale * dW2[i][j];
        }
        for (int i = 0; i < H3; i++) {
            b3[i] -= scale * db3[i];
            for (int j = 0; j < H2; j++) W3[i][j] -= scale * dW3[i][j];
        }
        for (int i = 0; i < K; i++) {
            b4[i] -= scale * db4[i];
            for (int j = 0; j < H3; j++) W4[i][j] -= scale * dW4[i][j];
        }
    }

    // Train: accepts training inputs X (list of double[d]) and labels (int[] 0..K-1)
    // Uses MSE (Mean Squared Error) as total training error
    // For each epoch, runs mini-batch gradient descent:
    //      1) shuffle training samples
    //      2) process in batches
    //      3) accumulate gradients
    //      4) update weights after each batch
    //  Prints the total MSE per epoch
    //  Stopping rule:
    //      Must run at least minEpochs
    //      Stop early if |E_epoch - E_prev| < stopDiffThreshold
    //      Or stop if epoch reaches maxEpochs
    public void train(List<double[]> X, int[] labels) {
        int N = X.size();
        if (N == 0) return;

        int epoch = 0;
        double prevTotalError = Double.POSITIVE_INFINITY;

        // prepare index array for shuffling
        Integer[] idx = new Integer[N];

        while (epoch < maxEpochs) {
            epoch++;
            // shuffle
            for (int i = 0; i < N; i++) idx[i] = i;
            List<Integer> idxList = Arrays.asList(idx);
            Collections.shuffle(idxList, rnd);
            idxList.toArray(idx);

            double totalError = 0.0;

            int processed = 0;
            while (processed < N) {
                int end = Math.min(processed + batchSize, N);
                int currentBatchSize = end - processed;

                // zero accumulators
                double[][] dW1 = new double[H1][d]; double[] db1 = new double[H1];
                double[][] dW2 = new double[H2][H1]; double[] db2 = new double[H2];
                double[][] dW3 = new double[H3][H2]; double[] db3 = new double[H3];
                double[][] dW4 = new double[K][H3]; double[] db4 = new double[K];

                // process batch
                for (int bi = processed; bi < end; bi++) {
                    int ii = idx[bi];
                    double[] x = X.get(ii);
                    int lab = labels[ii];

                    // forward
                    double[] y = forward(x);

                    double sampleLoss = 0.0;
                    for (int k = 0; k < K; k++) {
                        double t_k = (k == lab ? 1.0 : 0.0);
                        double diff = t_k - y[k];
                        sampleLoss += 0.5 * diff * diff;     // MSE
                    }
                    totalError += sampleLoss;

                    // build target one-hot vector t
                    double[] t = new double[K];
                    t[lab] = 1.0;

                    // compute gradients for this example into accumulators
                    computeGradientsSingle(x, t, dW1, db1, dW2, db2, dW3, db3, dW4, db4);
                }

                // apply gradients averaged over batch
                applyGradients(dW1, db1, dW2, db2, dW3, db3, dW4, db4, currentBatchSize);

                processed = end;
            } // end processing all batches

            // Print total training error for this epoch (sum over samples)
            System.out.printf("Epoch %d: Total training error = %.8f%n", epoch, totalError);

            // stopping condition: must run at least minEpochs
            if (epoch >= minEpochs) {
                double diff = Math.abs(totalError - prevTotalError);
                if (diff < stopDiffThreshold) {
                    System.out.printf("Stopping after %d epochs: |E - E_prev| = %.10f < %.10f%n",
                            epoch, diff, stopDiffThreshold);
                    break;
                }
            }

            prevTotalError = totalError;
        } // end epochs

        System.out.printf("Training finished after %d epochs.%n", Math.min(epoch, maxEpochs));
    }

    // Predict label index for one sample
    public int predict(double[] x) {
        double[] y = forward(x);
        int best = 0;
        double mv = y[0];
        for (int i = 1; i < K; i++) {
            if (y[i] > mv) { mv = y[i]; best = i; }
        }
        return best;
    }

    // Evaluate accuracy on dataset
    public double evaluateAccuracy(List<double[]> X, int[] labels) {
        int N = X.size();
        if (N == 0) return 0.0;
        int ok = 0;
        for (int i = 0; i < N; i++) {
            int p = predict(X.get(i));
            if (p == labels[i]) ok++;
        }
        return (double) ok / N;
    }

}
