package LossFunctions;

public interface LossFunction
{
    double Loss(double[] predicted, double[] target);
    double[] Derivate(double[] predicted, double[] target);
}
