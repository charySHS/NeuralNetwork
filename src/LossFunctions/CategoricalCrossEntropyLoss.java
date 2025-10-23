package LossFunctions;

public class CategoricalCrossEntropyLoss implements LossFunction
{
    @Override
    public double Loss(double[] predictions, double[] targets)
    {
        double eps = 1e-15;
        double loss = 0.0;

        for (int i = 0; i < predictions.length; i++)
        {
            double p = Math.max(eps, Math.min(1 - eps, predictions[i]));
            loss -= targets[i] * Math.log(p);
        }

        return loss;
    }

    @Override
    public double[] Derivate(double[] predictions, double[] targets)
    {
        double[] gradient = new double[predictions.length];

        for (int i = 0; i < predictions.length; i++) gradient[i] = predictions[i] - targets[i];

        return gradient;
    }
}
