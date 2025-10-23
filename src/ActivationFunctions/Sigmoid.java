package ActivationFunctions;

public class Sigmoid implements ActivationFunction
{
    @Override
    public double Activate(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    @Override
    public double Derivate(double x)
    {
        double fX = Activate(x);

        return fX * (1 - fX);
    }

    @Override
    public String GetName() { return "Sigmoid"; }

}
