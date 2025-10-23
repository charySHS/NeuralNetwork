package ActivationFunctions;

public class ReLU implements ActivationFunction
{
    @Override
    public double Activate(double x) { return Math.max(0, x); }

    @Override
    public double Derivate(double x) { return x > 0 ? 1.0 : 0.0; }

    @Override
    public String GetName() { return "Rectified Linear Unit"; }

}
