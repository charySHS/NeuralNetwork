package ActivationFunctions;

public class LeakyReLU implements ActivationFunction
{
    private final double alpha;

    public LeakyReLU() { this(0.01); }

    public LeakyReLU(double alpha) { this.alpha = alpha; }

    @Override
    public double Activate(double x) { return x > 0 ? x : alpha * x; }

    @Override
    public double Derivate(double x) { return x > 0 ? 1.0 : alpha; }

    @Override
    public String GetName() { return "LeakyReLU"; }
}
