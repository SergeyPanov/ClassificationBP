package cz.vut.fit.network;

import cz.vut.fit.neuron.HiddenNeuron;
import cz.vut.fit.neuron.InputNeuron;
import cz.vut.fit.neuron.Neuron;
import cz.vut.fit.neuron.OutputNeuron;
import cz.vut.fit.synapse.Synapse;


/**
 * Network class represent network with single layer of hidden neurons.
 */
public class Network {

    /**
     * Layers.
     */
    private InputNeuron[] inputNeurons;
    private HiddenNeuron[] hiddenNeurons;
    private OutputNeuron[] outputNeurons;

    /**
     * The learning rate.
     */
    private final double learnRate;

    /**
     * The momentum.
     */
    private final double momentum;

    /**
     * The global error.
     */
    private double globalError;

    public Network(int inputCount,
                   int hiddenCount,
                   int outputCount,
                   double learnRate,
                   double momentum){
        int i;

        this.learnRate = learnRate;
        this.momentum = momentum;

        this.inputNeurons = new InputNeuron[inputCount];
        for ( i = 0 ; i < inputCount; ++ i ){
            inputNeurons[i] = new InputNeuron(0);
        }

        this.hiddenNeurons = new HiddenNeuron[hiddenCount];
        double hiddenLayerBIAS = 1 - (Math.random());
        for ( i = 0;  i < hiddenCount; ++ i){
            hiddenNeurons[i] = new HiddenNeuron();
            hiddenNeurons[i].setBias(hiddenLayerBIAS);
        }

        double outputLayerBIAS = 1 - (Math.random());

        this.outputNeurons = new OutputNeuron[outputCount];
        for ( i = 0 ; i < outputCount; ++ i ){
            outputNeurons[i] = new OutputNeuron();
            outputNeurons[i].setBias(outputLayerBIAS);
        }
    }

    /**
     * Count RootMSE
     * @return RootMSE
     */
    public double getError(int len){
        double err = Math.sqrt(globalError / len * outputNeurons.length);
        globalError = 0; // clear the accumulator
        return err;
    }

    /**
     * Calculate outputs based on input.
     * @param input input of the neural network.
     * @return values of the hidden layer.
     */
    public double[] calculateOutputs(Double input[]){

        for (int i = 0; i < input.length; ++ i){
            inputNeurons[i].setFire(input[i]);
        }

        for (HiddenNeuron hiddenNeuron :
                hiddenNeurons) {
            hiddenNeuron.calculateFile();
        }

        int i = 0;

        double[] result = new double[outputNeurons.length];

        for (OutputNeuron outputNeuron:
             outputNeurons) {
            outputNeuron.calculateFile();
            result[i++] = outputNeuron.getFire();
        }

        return result;
    }


    /**
     * Count sigmas, gradients, deltas and adjust weights.
     * @param ideal ideal result.
     */
    public void learn(Double ideal[]){
        /*
        Calculate sigma for output neurons.
         */
        for (int i = 0 ; i < outputNeurons.length; ++ i){

            globalError += (ideal[i] - outputNeurons[i].getFire()) * (ideal[i] - outputNeurons[i].getFire());

            double outNeuronDerivation = outputNeurons[i].derivation();

            outputNeurons[i].setSigma((ideal[i] - outputNeurons[i].getFire()) * outNeuronDerivation );
        }

        /*
        Hidden layer calculation
         */
        for (HiddenNeuron hiddenNeuron:
             hiddenNeurons) {
            double hiddenNeuronDerivation = hiddenNeuron.derivation();

            double sum = hiddenNeuron.getOutputSynapses().stream().mapToDouble(synapse -> synapse.getWeight() * synapse.getTo().getSigma()).sum();

            hiddenNeuron.setSigma(sum * hiddenNeuronDerivation);

            /*
            Calculate gradients for each output synapse.
             */
            for (Synapse outSynapse:
                 hiddenNeuron.getOutputSynapses()) {
                outSynapse.setGrad(hiddenNeuron.getFire() * outSynapse.getTo().getSigma());
            }
            /*
            Calculate delta W for each synapse.
             */
            for (Synapse outSynapse:
                 hiddenNeuron.getOutputSynapses()) {
                double deltaW = learnRate * outSynapse.getGrad() + momentum * outSynapse.getOldDeltaWeight();
                outSynapse.adjustWeight(deltaW);
            }
        }


        /*
        Input layer calculation.
         */

        for (InputNeuron inputNeuron:
             inputNeurons) {
            for (Synapse synapse:
                 inputNeuron.getOutputSynapses()) {
                synapse.setGrad(inputNeuron.getFire() * synapse.getTo().getSigma());
                double deltaW = learnRate * synapse.getGrad() + momentum * synapse.getOldDeltaWeight();
                synapse.adjustWeight(deltaW);
            }
        }

    }

    /**
     * Connect two layers by synapses.
     * @param layer1 "From" layer.
     * @param layer2 "To" layer.
     */

    private void connectLayers(Neuron[] layer1, Neuron[] layer2){
        for (Neuron neuron1:
             layer1) {

            for (Neuron neuron2:
                 layer2) {

                Synapse synapse = new Synapse();

                neuron1.addOutputSynapse(synapse);

                synapse.setFrom(neuron1);

                neuron2.addInputSynapse(synapse);

                synapse.setTo(neuron2);
            }

        }
    }

    /**
     * Reset the network.
     */
    public void reset(){
        /*
        Connect input neurons and hidden neurons.
         */
        connectLayers(inputNeurons, hiddenNeurons);
        connectLayers(hiddenNeurons, outputNeurons);

    }

}
