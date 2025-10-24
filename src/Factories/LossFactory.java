package Factories;

import LossFunctions.*;

public class LossFactory
{
    public static LossFunction FromName(String name)
    {
        return switch (name)
        {
            case "CategoricalCrossEntropyLoss" ->  new CategoricalCrossEntropyLoss();
            case "MeanSquaredError" ->  new MeanSquaredError();
            default ->  throw new  IllegalArgumentException("Unknown Loss Function: " + name);
        };
    }
}
