package ActivationFunctions;

public class Softmax implements ActivationFunction
{
    private double[] LastOutput;

    @Override
    public double Activate(double x) { return x; }

    @Override
    public double Derivate(double x) { return 1; }

    @Override
    public String GetName() { return "Softmax"; }

    public double[] ActivateLayer(double[] z)
    {
        double max = Double.NEGATIVE_INFINITY;

        for (double value : z) max = Math.max(max, value);

        double sum = 0.0;
        double[] exp = new double[z.length];

        for (int i = 0; i < z.length; i++)
        {
            exp[i] = Math.exp(z[i] - max);
            sum += exp[i];
        }

        LastOutput = new double[z.length];

        for (int i = 0; i < z.length; i++) LastOutput[i] = exp[i] / sum;

        return LastOutput;
    }

    public double[] GetLastOutput() { return LastOutput; }
}
