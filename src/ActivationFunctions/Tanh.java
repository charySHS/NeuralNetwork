package ActivationFunctions;

public class Tanh implements ActivationFunction
{
    @Override
    public double Activate(double x) { return Math.tanh(x); }

    @Override
    public double Derivate(double x)
    {
        double t = Math.tanh(x);

        return 1 - t * t;
    }

    @Override
    public String GetName() { return "Tanh"; }

}
