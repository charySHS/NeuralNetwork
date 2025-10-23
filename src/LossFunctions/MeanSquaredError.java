package LossFunctions;

public class MeanSquaredError implements LossFunction
{
    @Override
    public double Loss(double[] predicted, double[] target)
    {
        double sum = 0;

        for (int i = 0; i < predicted.length; i++)
        {
            double difference = predicted[i] - target[i];
            sum += difference * difference;
        }

        return sum / predicted.length;
    }

    @Override
    public double[] Derivate(double[] predicted, double[] target)
    {
        double[] gradient = new double[predicted.length];

        for (int i = 0; i < predicted.length; i++) { gradient[i] = 2 * (predicted[i] - target[i]) / predicted.length; }

        return gradient;
    }
}
