package Neurons;

import ActivationFunctions.*;
import Network.OptimizationType;

import java.util.Random;

public class Neuron
{
    // -------------------------------------------------------------------------------------------------------
    // -- Variables
    // -------------------------------------------------------------------------------------------------------
    private final ActivationFunction ActivationFunction;
    private final OptimizationType Optimizer;

    private double[] Weights;

    private double Bias;
    private double Output;
    private double LastZ;

    private final double LearningRate;
    private final double Beta1;
    private final double Beta2;
    private final double Decay;

    // For Adam
    private double[] MomentaryWeights;
    private double[] VectoredWeights;

    private double MomentaryBias;
    private double VectoredBias;

    // -------------------------------------------------------------------------------------------------------
    // -- Functions
    // -------------------------------------------------------------------------------------------------------
    public Neuron(int inputCount, ActivationFunction activation, OptimizationType optimizationType, double learningRate, double beta1, double beta2, double decay)
    {
        ActivationFunction = activation;
        Optimizer = optimizationType;
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Decay = decay;

        Weights = new double[inputCount];
        MomentaryWeights = new double[inputCount];
        VectoredWeights = new double[inputCount];

        Bias = 0.0;
        MomentaryBias = 0.0;
        VectoredBias = 0.0;

        InitializeWeights(inputCount);
    }

    public double FeedForward(double[] inputs)
    {
        double sum = Bias;

        for (int i = 0; i < Weights.length; i++) sum += Weights[i] * inputs[i];

        LastZ = sum;
        Output = ActivationFunction.Activate(sum);

        return Output;
    }

    public void UpdateWeights(double[] inputs, double error, int t)
    {
        double learningRateT = LearningRate / (1.0 + Decay * t);

        for (int i = 0; i < Weights.length; i++)
        {
            double gradient = error * inputs[i];

            if (Optimizer == Optimizer.Adam)
            {
                MomentaryWeights[i] = Beta1 * MomentaryWeights[i] + (1 - Beta1) * gradient;
                VectoredWeights[i] = Beta2 * VectoredWeights[i] + (1 - Beta2) * gradient * gradient;

                double momentaryHat = MomentaryWeights[i] / (1 - Math.pow(Beta1, t));
                double vectoredHat = VectoredWeights[i] / (1 - Math.pow(Beta2, t));

                Weights[i] -= learningRateT * momentaryHat / (Math.sqrt(vectoredHat) + 1e-8);
            }
            else { Weights[i] -= learningRateT * gradient; }
        }

        double gradientBias = error;

        if (Optimizer == Optimizer.Adam)
        {
            MomentaryBias = Beta1 * MomentaryBias + (1 - Beta1) * gradientBias;
            VectoredBias = Beta2 * VectoredBias + (1 - Beta2) * gradientBias * gradientBias;

            double momentaryHatBias = MomentaryBias / (1 - Math.pow(Beta1, t));
            double vectoredHatBias = VectoredBias / (1 - Math.pow(Beta2, t));

            Bias -= learningRateT * momentaryHatBias / (Math.sqrt(vectoredHatBias) + 1e-8);
        }
        else { Bias -= learningRateT * gradientBias; }
    }

    // -------------------------------------------------------------------------------------------------------
    // -- Helper Functions
    // -------------------------------------------------------------------------------------------------------

    /**
     * Function for initializing weights based on ActivationFunction type
     *
     * @param inputCount Number of neurons to initialize.
     */
    private void InitializeWeights(int inputCount)
    {
        Random random = new  Random();
        double standardDeviation;

        if (ActivationFunction instanceof ReLU || ActivationFunction instanceof LeakyReLU) { standardDeviation = Math.sqrt(2.0 / inputCount); }     // HE Initialization
        else if (ActivationFunction instanceof Sigmoid || ActivationFunction instanceof Tanh) { standardDeviation = Math.sqrt(1.0 / inputCount); }  // Xavier Initialize
        else if (ActivationFunction instanceof Softmax) { standardDeviation = Math.sqrt(1.0 / (inputCount + 1)); }
        else { standardDeviation = 0.01; }

        for (int i = 0; i < inputCount; i++) Weights[i] = random.nextGaussian() * standardDeviation;
        Bias = 0.0;
    }

    public double[] GetWeights() { return Weights; }

    public double GetLastZ() { return LastZ; }

    public double GetOutput() { return Output; }

    public double GetBias() { return Bias; }

    public ActivationFunction GetActivationFunction() { return ActivationFunction; }

    public void SetWeights(double[] weights) { Weights = weights.clone(); }

    public void SetBias(double bias) { Bias = bias; }
}
