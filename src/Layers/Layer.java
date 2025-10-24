package Layers;

import ActivationFunctions.*;
import Network.NeuralNetwork;
import Network.OptimizationType;
import Network.WeightInitialization;
import Neurons.Neuron;

import java.io.OutputStream;
import java.util.Random;

public class Layer
{
    // -------------------------------------------------------------------------------------------------------
    // -- Variables
    // -------------------------------------------------------------------------------------------------------

    private final ActivationFunction ActivationFunction;

    private final OptimizationType Optimizer;

    private final Neuron[] Neurons;

    private double[] Outputs;

    private final int InputSize;
    private final int OutputSize;

    private final double Beta1;
    private final double Beta2;
    private final double Decay;
    private final double LearningRate;

    private final boolean IsSoftmax;

    // -------------------------------------------------------------------------------------------------------
    // -- Functions
    // -------------------------------------------------------------------------------------------------------

    public Layer(ActivationFunction activationFunction, OptimizationType optimizer, double learningRate, double beta1, double beta2, double decay, int inputSize, int outputSize)
    {
        ActivationFunction = activationFunction;
        Optimizer = optimizer;
        IsSoftmax = ActivationFunction instanceof Softmax;

        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Decay = decay;

        InputSize = inputSize;
        OutputSize = outputSize;

        Neurons = new Neuron[OutputSize];

        for (int i = 0; i < outputSize; i++) Neurons[i] = new Neuron(InputSize, ActivationFunction, Optimizer, LearningRate, Beta1, Beta2, Decay);

        Outputs = new double[OutputSize];
    }

    public double[] FeedForward(double[] inputs)
    {
        double[] z = new double[OutputSize];

        for (int i = 0; i < OutputSize; i++) z[i] = Neurons[i].FeedForward(inputs);

        if (IsSoftmax) Outputs = ((Softmax) ActivationFunction).ActivateLayer(z);
        if (!IsSoftmax) Outputs = z;

        return Outputs;
    }

    public double[] Backpropagate(double[] inputs, double[] errors, int t)
    {
        double[] previousErrors = new double[InputSize];

        for (int i = 0; i < OutputSize; i++)
        {
            double delta = errors[i];

            if (!IsSoftmax) delta *= Neurons[i].GetActivationFunction().Derivate(Neurons[i].GetLastZ());

            double[] weights = Neurons[i].GetWeights();

            for (int j = 0; j < InputSize; j++) previousErrors[j] += delta * weights[j];

            Neurons[i].UpdateWeights(inputs, delta, t);
        }

        return previousErrors;
    }

    // -------------------------------------------------------------------------------------------------------
    // -- Helper Functions
    // -------------------------------------------------------------------------------------------------------

    public double[] GetOutput() { return Outputs; }

    public Neuron[] GetNeurons() { return Neurons; }

    public int GetInputSize() { return InputSize; }

    public int GetOutputSize() { return OutputSize; }

    public ActivationFunction GetActivationFunction() { return ActivationFunction; }

    public double[][] GetWeights()
    {
        double[][] weights = new double[Neurons.length][];

        for (int i = 0; i < Neurons.length; i++) weights[i] = Neurons[i].GetWeights().clone();

        return weights;
    }

    public double[] GetBiases()
    {
        double[] biases = new double[Neurons.length];

        for (int i = 0; i < Neurons.length; i++) biases[i] = Neurons[i].GetBias();

        return biases;
    }

    public void SetWeights(double[][] newWeights) { for (int i = 0; i < Neurons.length; i++) Neurons[i].SetWeights(newWeights[i]); }

    public void SetBiases(double[] newBiases) { for (int i = 0; i < Neurons.length; i++) Neurons[i].SetBias(newBiases[i]); }


}
