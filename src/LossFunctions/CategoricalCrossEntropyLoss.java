package LossFunctions;

public class CategoricalCrossEntropyLoss implements LossFunction
{
    private static final double Epsilon = 1e-12;

    @Override
    public double Loss(double[] target, double[] predicted)
    {
        double sum = 0.0;

        for (int i = 0; i < predicted.length; i++)
        {
            double p = Math.max(Epsilon, Math.min(1.0 - Epsilon, predicted[i]));
            sum += target[i] * Math.log(p);
        }

        return sum / predicted.length;
    }

    @Override
    public double[] Derivate(double[] target, double[] predicted)
    {
        double[] gradient = new double[predicted.length];

        for (int i = 0; i < predicted.length; i++) gradient[i] = predicted[i] - target[i];

        return gradient;
    }
}
