package cz.vut.fit.network;

import cz.vut.fit.neuron.HiddenNeuron;
import cz.vut.fit.neuron.InputNeuron;
import cz.vut.fit.neuron.Neuron;
import cz.vut.fit.neuron.OutputNeuron;
import cz.vut.fit.synapse.Synapse;

import java.io.Serializable;


/**
 * Network class represent network with single layer of hidden neurons.
 */
public class Network implements Serializable {

    /**
     * Input neurons
     */
    private InputNeuron[] inputNeurons;
    /**
     * Hidden neurons
     */
    private HiddenNeuron[] hiddenNeurons;
    /**
     * Output neurons
     */
    private OutputNeuron[] outputNeurons;

    /**
     * The learning rate.
     */
    private final Double learnRate;

    /**
     * The momentum.
     */
    private final Double momentum;

    /**
     * The global error.
     */
    private Double globalError;

    /**
     * Initialize neural network with parameters:
     * @param inputCount    Number of neurons at the input layer.
     * @param hiddenCount   Number of neurons at the hidden layer.
     * @param outputCount   Number of neurons at the output layer.
     * @param learnRate     The learning rate value(default 0.7).
     * @param momentum      The momentum value(default 0.3).
     * @param hiddenLayerBias   The BIAS value for hidden layer(default 0.7).
     * @param outputLayerBias   The BIAS value for output layer(default 0.7).
     */
    public Network(int inputCount,
                   int hiddenCount,
                   int outputCount,
                   Double learnRate,
                   Double momentum,
                   Double hiddenLayerBias,
                   Double outputLayerBias
                   ){
        int i;

        globalError = 0.0;

        this.learnRate = learnRate;
        this.momentum = momentum;

        this.inputNeurons = new InputNeuron[inputCount];
        for ( i = 0 ; i < inputCount; ++ i ){
            inputNeurons[i] = new InputNeuron(0);
        }

        this.hiddenNeurons = new HiddenNeuron[hiddenCount];
        for ( i = 0;  i < hiddenCount; ++ i){
            hiddenNeurons[i] = new HiddenNeuron();
            hiddenNeurons[i].setBias(hiddenLayerBias);
        }

        this.outputNeurons = new OutputNeuron[outputCount];
        for ( i = 0 ; i < outputCount; ++ i ){
            outputNeurons[i] = new OutputNeuron();
            outputNeurons[i].setBias(outputLayerBias);
        }
    }

    /**
     * Calculate and return RootMSE.
     * @return RootMSE
     */
    public Double getError(int len){
        Double err = Math.sqrt(globalError / len * outputNeurons.length);
        globalError = 0.0; // clear the accumulator
        return err;
    }

    /**
     * Calculate outputs based on input.
     * @param input input of the neural network.
     * @return values from output neurons.
     */
    public Double[] calculateOutputs(Double[] input){

        for (int i = 0; i < input.length; ++ i){
            inputNeurons[i].setFire(input[i]);
        }

        for (HiddenNeuron hiddenNeuron :
                hiddenNeurons) {
            hiddenNeuron.calculateFile();
        }

        int i = 0;

        Double[] result = new Double[outputNeurons.length];

        for (OutputNeuron outputNeuron:
             outputNeurons) {
            outputNeuron.calculateFile();
            result[i++] = outputNeuron.getFire();
        }

        return result;
    }


    /**
     * Count gradients, deltas and adjust weights.
     * @param ideal ideal results.
     */
    public void learn(Double ideal[]){
        /*
        Output layer calculation
         */
        for (int i = 0 ; i < outputNeurons.length; ++ i){

            globalError += (ideal[i] - outputNeurons[i].getFire()) * (ideal[i] - outputNeurons[i].getFire());

            Double outNeuronDerivation = outputNeurons[i].derivation();

            outputNeurons[i].setDelta((ideal[i] - outputNeurons[i].getFire()) * outNeuronDerivation );
        }

        /*
        Hidden layer calculation
         */
        for (HiddenNeuron hiddenNeuron:
             hiddenNeurons) {
            Double hiddenNeuronDerivation = hiddenNeuron.derivation();

            Double sum = hiddenNeuron.getOutputSynapses().stream().mapToDouble(synapse -> synapse.getWeight() * synapse.getTo().getDelta()).sum();

            hiddenNeuron.setDelta(sum * hiddenNeuronDerivation);

            /*
            Calculate gradients for each output synapse.
             */
            for (Synapse outSynapse:
                 hiddenNeuron.getOutputSynapses()) {
                outSynapse.setGrad(hiddenNeuron.getFire() * outSynapse.getTo().getDelta());
            }
            /*
            Calculate delta W for each synapse.
             */
            for (Synapse outSynapse:
                 hiddenNeuron.getOutputSynapses()) {
                Double deltaW = learnRate * outSynapse.getGrad() + momentum * outSynapse.getOldDeltaWeight();
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
                synapse.setGrad(inputNeuron.getFire() * synapse.getTo().getDelta());
                Double deltaW = learnRate * synapse.getGrad() + momentum * synapse.getOldDeltaWeight();
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
     * Invoke 'connectionLayers' method for connection input layer with hidden and hidden with output.
     */
    public void reset(){
        /*
        Connect input neurons and hidden neurons.
         */
        connectLayers(inputNeurons, hiddenNeurons);
        connectLayers(hiddenNeurons, outputNeurons);

    }

}
