public class Activation {

    public static double activate(String type, double u) {
        switch (type.toLowerCase()) {
            case "logistic":
                return 1.0 / (1.0 + Math.exp(-u));
            case "tanh":
                return Math.tanh(u);
            case "relu":
            default:
                return Math.max(0.0, u);
        }
    }

    public static double derivative(String type, double u, double activatedValue) {
        switch (type.toLowerCase()) {
            case "logistic":
                return activatedValue * (1.0 - activatedValue);
            case "tanh":
                return 1.0 - activatedValue * activatedValue;
            case "relu":
            default:
                return u > 0.0 ? 1.0 : 0.0;
        }
    }
}
