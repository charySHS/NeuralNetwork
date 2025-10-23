package LossFunctions;

public class CrossEntropyLoss implements LossFunction
{
    private static final double Epsilon = 1e-9;

    @Override
    public double Loss(double[] predicted, double[] target)
    {
        double sum = 0;

        for (int i = 0; i < predicted.length; i++)
        {
            double p = Math.min(Math.max(predicted[i], Epsilon), 1.0 - Epsilon);
            sum -= (target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p));
        }

        return sum / predicted.length;
    }

    @Override
    public double[] Derivate(double[] predicted, double[] target)
    {
        double[] gradient = new double[predicted.length];

        for (int i = 0; i < predicted.length; i++)
        {
            double p = Math.min(Math.max(predicted[i], Epsilon), 1.0 - Epsilon);
            gradient[i] = (p - target[i]) / ((p) * (1 - p));

            if (gradient[i] > 10.0) gradient[i] = 10.0;
            else if (gradient[i] < -10.0) gradient[i] = -10.0;
        }

        return gradient;
    }
}
