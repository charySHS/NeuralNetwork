package Layers;

import ActivationFunctions.*;
import Network.NeuralNetwork;
import Network.OptimizationType;
import Network.WeightInitialization;
import Neurons.Neuron;

import java.util.Random;

public class Layer
{
    // -------------------------------------------------------------------------------------------------------
    // -- Variables
    // -------------------------------------------------------------------------------------------------------

    private final ActivationFunction ActivationFunction;

    private final OptimizationType OptimizationType;

    private final Neuron[] Neurons;

    private double[] Outputs;

    private final int InputSize;
    private final int OutputSize;

    private final double Beta1;
    private final double Beta2;
    private final double Decay;
    private final double LearningRate;

    private boolean IsSoftmax;

    // -------------------------------------------------------------------------------------------------------
    // -- Functions
    // -------------------------------------------------------------------------------------------------------
    public Layer(ActivationFunction activationFunction, OptimizationType optimizationType, double learningRate, double beta1, double beta2, double decay, int inputSize, int outputSize)
    {
        ActivationFunction = activationFunction;
        OptimizationType = optimizationType;

        LearningRate = learningRate;
        InputSize = inputSize;
        OutputSize = outputSize;
        Decay = decay;
        Beta1 = beta1;
        Beta2 = beta2;

        IsSoftmax = ActivationFunction instanceof Softmax;

        Neurons = new Neuron[outputSize];

        for (int i = 0; i < outputSize; i++) Neurons[i] = new Neuron(inputSize, activationFunction, optimizationType, learningRate, beta1, beta2, decay);

        Outputs = new double[outputSize];
    }

    public double[] FeedForward(double[] inputs)
    {
        double[] zValues = new double[OutputSize];

        for (int i = 0; i < OutputSize; i++) zValues[i] = Neurons[i].FeedForward(inputs);

        if (IsSoftmax) Outputs = ((Softmax) ActivationFunction).ActivateLayer(zValues);
        else Outputs = zValues;

        return Outputs;
    }

    public double[] Backpropogate(double[] inputs, double[] errors, int t)
    {
        double[] propagatedErrors = new double[InputSize];

        for (int i = 0; i < OutputSize; i++)
        {
            Neurons[i].UpdateWeights(inputs, errors[i], t);

            for (int j = 0; j < propagatedErrors.length; j++) propagatedErrors[j] += errors[i] * Neurons[i].GetWeights()[j];
        }

        return propagatedErrors;
    }

    // -------------------------------------------------------------------------------------------------------
    // -- Helper Functions
    // -------------------------------------------------------------------------------------------------------

    public double[] GetOutputs() { return Outputs; }

    public int GetOutputSize() { return OutputSize; }

    public int GetInputSize() { return InputSize; }

    public Neuron[] GetNeurons() { return Neurons; }

}
