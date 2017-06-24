import java.util.Random;


public class MainClass {

    public final Matrix input = new Matrix(2, 1);
    public static final double TARGET = 0;

    Matrix weight0 = new Matrix(3, 2);
    Matrix hiddenToOutputWeight = new Matrix(3, 1);

    public static void main(String[] args){
        MainClass mainClass = new MainClass();
    }

    MainClass(){
        input.set(1, 1, 1);
        input.set(2, 1, 1);

        generateWeight(3, 2, weight0);
        generateWeight(3, 1, hiddenToOutputWeight);

        Matrix output = new Matrix(1, 1);

        for (int i = 0; i < 1000; i++) {
            Matrix hiddenNodes1 = weight0.mul(input);
            System.out.println("hiddenNodes1");
            hiddenNodes1.println();
            Matrix hiddenNodesAfterSigmoid = multiplyByS(hiddenNodes1);
            hiddenNodesAfterSigmoid = hiddenNodesAfterSigmoid.transpose();
            System.out.println("hiddenNodesAfterSigmoid");
            hiddenNodesAfterSigmoid.println();
            System.out.println();
            if (i > 0){
                output = hiddenNodesAfterSigmoid.mul(hiddenToOutputWeight.transpose());
            } else {
                output = hiddenNodesAfterSigmoid.mul(hiddenToOutputWeight);
            }
            output.println();
            System.out.println();
            hiddenToOutputWeight.println();
            double resultOfForwardPropagation = S(output.get(1, 1));
            System.out.println("f " + resultOfForwardPropagation);

            double marginOfError = marginOfError(output.get(1, 1));
            double deltaOutputSum = derivativeS(output.get(1, 1)) * marginOfError;

            Matrix deltaWeights1 = hiddenNodesAfterSigmoid.mul(deltaOutputSum);
            if (i > 0) {
                hiddenToOutputWeight = hiddenToOutputWeight.add(deltaWeights1);
            } else {
                hiddenToOutputWeight = hiddenToOutputWeight.transpose().add(deltaWeights1);
            }
            Matrix hiddenNodesAfterDerivativeS = new Matrix(3, 1);
            Matrix deltaHiddenSum1 = new Matrix(3, 1);
            for (int counter = 1; counter <= 3; counter++) {
                hiddenNodesAfterDerivativeS.set(counter, 1, derivativeS(hiddenNodes1.get(counter, 1)));
                deltaHiddenSum1.set(counter, 1, deltaOutputSum * hiddenToOutputWeight.transpose().get(counter, 1) * hiddenNodesAfterDerivativeS.get(counter, 1));
            }

            Matrix deltaWeightsHiddenToOutput = deltaHiddenSum1.mul(input.transpose());
            weight0 = weight0.add(deltaWeightsHiddenToOutput);

            weight0.println();
        }

        System.out.println("\n" + "This is the final output " + output.get(1, 1));
    }

    public void generateWeight(int m, int n, Matrix matrix){
        Random random = new Random();
        for (int counter = 1; counter <= m; counter++) {
            for (int counter1 = 1; counter1 <= n; counter1++) {
                matrix.set(counter, counter1, Math.max(0, Math.min(1, (random.nextGaussian() / 4 + 0.5))));
            }
        }
    }

    public Matrix multiplyByS(Matrix matrix){
        Matrix result = new Matrix(matrix.rows(), matrix.cols());
        for (int counter = 1; counter <= matrix.rows(); counter++){
            result.set(counter, 1, S(matrix.get(counter, 1)));
        }
        return result;
    }

    public double S(double x){
        return 1/(1 + Math.pow(Math.E, (-x)));
    }

    public double marginOfError(double a){
        return TARGET - a;
    }

    public double derivativeS(double x){
        return (Math.pow(Math.E, -x)) / Math.pow((1 + Math.pow(Math.E, -x)), 2);
    }
}
