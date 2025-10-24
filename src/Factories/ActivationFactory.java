package Factories;

import ActivationFunctions.*;

public class ActivationFactory
{
    public static ActivationFunction FromName(String name)
    {
        return switch (name)
        {
            case "ReLU" -> new ReLU();
            case "LeakyReLU" -> new LeakyReLU();
            case "Sigmoid" -> new Sigmoid();
            case "Tanh" -> new Tanh();
            case "Softmax" -> new Softmax();
            default -> throw new IllegalArgumentException("Invalid activation function name: " + name);
        };
    }
}
