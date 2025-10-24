package Network;

import java.io.FileWriter;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import ActivationFunctions.*;
import Factories.*;
import Layers.Layer;
import LossFunctions.*;

public class NeuralNetwork
{
    // -------------------------------------------------------------------------------------------------------
    // -- Variables
    // -------------------------------------------------------------------------------------------------------

    private final Layer[] Layers;

    private final LossFunction LossFunction;

    private final OptimizationType Optimizer;

    private final double LearningRate;
    private final double Beta1;
    private final double Beta2;
    private final double Decay;

    // -------------------------------------------------------------------------------------------------------
    // -- Functions
    // -------------------------------------------------------------------------------------------------------

    public NeuralNetwork(LossFunction lossFunction, OptimizationType optimizationType, double learningRate, double beta1, double beta2, double decay, ActivationFunction[] hiddenActivations, int... layerSizes)
    {
        if (layerSizes.length < 2) throw new IllegalArgumentException("Network must have at least input and output layer.");

        LossFunction = lossFunction;
        Optimizer = optimizationType;

        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Decay = decay;

        int numLayers = layerSizes.length - 1;
        Layers = new Layer[numLayers];

        if (hiddenActivations.length != numLayers -1) throw new IllegalArgumentException("Number of hidden activation functions must match hidden layer count.");

        for (int i = 0; i < numLayers - 1; i++) Layers[i] = new Layer(hiddenActivations[i], Optimizer, LearningRate, Beta1, Beta2, Decay, layerSizes[i], layerSizes[i + 1]);

        ActivationFunction outputActivation = (LossFunction instanceof CategoricalCrossEntropyLoss) ? new Softmax() : new Sigmoid();
        Layers[numLayers - 1] = new Layer(outputActivation, Optimizer, LearningRate, Beta1, Beta2, Decay, layerSizes[numLayers - 1],  layerSizes[numLayers]);
    }

    public double[] Predict(double[] inputs)
    {
        double[] outputs = inputs;

        for (Layer layer : Layers) outputs = layer.FeedForward(outputs);

        return outputs;
    }

    public void TrainMiniBatch(double[][] inputs, double[][] targets, int batchSize, int epochs)
    {
        int t = 1;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double totalLoss = 0.0;

            for (int i = 0; i < inputs.length; i++)
            {
                double[] predicted = Predict(inputs[i]);
                totalLoss += LossFunction.Loss(targets[i], predicted);

                double[] errors = LossFunction.Derivate(targets[i], predicted);

                for (int l = Layers.length - 1; l >= 0; l--)
                {
                    double[] layerInputs = (l == 0) ? inputs[i] : Layers[l - 1].GetOutput();
                    errors = Layers[l].Backpropagate(layerInputs, errors, t);
                }

                t++;
            }

            double averageLoss = totalLoss / inputs.length;

            if (epoch % 100 == 0)
            {
                double currentLearningRate = LearningRate / (1.0 + Decay * epoch);
                System.out.printf("Epoch %d, Learning Rate=%.6f, Loss=%.6f%n", epoch, currentLearningRate, averageLoss);
            }
        }
    }

    // -------------------------------------------------------------------------------------------------------
    // -- Functions
    // -------------------------------------------------------------------------------------------------------

    public Layer[] GetLayers() { return Layers; }

    public double GetLearningRate() { return LearningRate; }

    public double GetBeta1() { return Beta1; }

    public double GetBeta2() { return Beta2; }

    public double GetDecay() { return Decay; }

}

