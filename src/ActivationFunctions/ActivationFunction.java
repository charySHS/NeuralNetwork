package ActivationFunctions;

public interface ActivationFunction
{
    public double Activate(double x);

    public double Derivate(double x);

    public String GetName();
}
