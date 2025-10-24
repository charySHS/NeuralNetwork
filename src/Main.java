import Factories.ActivationFactory;
import LossFunctions.*;
import ActivationFunctions.*;
import Network.*;

import ModelFunctions.*;

import java.io.IOException;

public class Main
{
    public static void main(String[] args) throws IOException {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}};

        ActivationFunction[] activations = new ActivationFunction[]{new ReLU(), new ReLU(), new LeakyReLU()};

        NeuralNetwork neuralNetwork = new NeuralNetwork(new CategoricalCrossEntropyLoss(), OptimizationType.Adam, 0.04, 0.9, 0.999, 0.00005, activations, 2, 24, 24, 24, 3);

        neuralNetwork.TrainMiniBatch(inputs, targets, 4, 1000);

        System.out.println("\nTesting (new model):");

        for (double[] input : inputs)
        {
            double[] output = neuralNetwork.Predict(input);
            System.out.printf("Input: [%d, %d] -> [%.3f, %.3f, %.3f]%n", (int) input[0], (int) input[1], output[0], output[1], output[2]);
        }
    }
}
