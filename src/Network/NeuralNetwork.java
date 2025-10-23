package Network;

import ActivationFunctions.*;
import Layers.Layer;
import LossFunctions.*;

public class NeuralNetwork
{
    // -------------------------------------------------------------------------------------------------------
    // -- Variables
    // -------------------------------------------------------------------------------------------------------

    private final Layer[] Layers;
    private final LossFunction LossFunction;
    private final OptimizationType OptimizationType;

    private final double LearningRate;
    private final double Beta1;
    private final double Beta2;
    private final double Decay;

    // -------------------------------------------------------------------------------------------------------
    // -- Functions
    // -------------------------------------------------------------------------------------------------------

    public NeuralNetwork(LossFunction lossFunction, OptimizationType optimizationType, double learningRate, double beta1, double beta2, double decay, ActivationFunction[] hiddenActivations, int... layerSizes)
    {
        if (layerSizes.length < 2) throw new IllegalArgumentException("Network must have at least an input and output layer.");

        LossFunction = lossFunction;
        OptimizationType = optimizationType;
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Decay = decay;

        int numLayers = layerSizes.length - 1;
        Layers = new Layer[numLayers];

        if (hiddenActivations.length != numLayers -1) throw new IllegalArgumentException("Number of hidden activations must match hidden layer size.");

        for (int i = 0; i < numLayers -1; i++) { Layers[i] = new Layer(hiddenActivations[i], optimizationType, learningRate, beta1, beta2, decay, layerSizes[i], layerSizes[i + 1]); }

        ActivationFunction outputActivation = (lossFunction instanceof CategoricalCrossEntropyLoss) ? new Softmax() : new Sigmoid();

        Layers[numLayers - 1] = new Layer(outputActivation, optimizationType, learningRate, beta1, beta2, decay, layerSizes[numLayers - 1], layerSizes[numLayers]);
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
                double[] output = Predict(inputs[i]);
                totalLoss += LossFunction.Loss(targets[i], output);

                double[] errors = LossFunction.Derivate(targets[i], output);

                double[] currentErrors = errors;

                for (int layerIndex = Layers.length - 1; layerIndex >= 0; layerIndex--)
                {
                    double[] layerInputs = (layerIndex == 0) ? inputs[i] : Layers[layerIndex - 1].GetOutputs();
                    currentErrors = Layers[layerIndex].Backpropogate(layerInputs, currentErrors, t);
                }

                t++;
            }

            double averageLoss = totalLoss / inputs.length;

            if (epoch % 100 == 0)
            {
                double currentLearningRate = LearningRate / (1.0 + Decay * epoch);
                System.out.printf("Epoch %d, Learning Rate=%.5f, Loss=%.6f%n", epoch, currentLearningRate, averageLoss);
            }
        }
    }
}

