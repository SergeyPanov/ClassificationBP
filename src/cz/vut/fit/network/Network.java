package cz.vut.fit.network;

import cz.vut.fit.neuron.HiddenNeuron;
import cz.vut.fit.neuron.InputNeuron;
import cz.vut.fit.neuron.Neuron;
import cz.vut.fit.neuron.OutputNeuron;
import cz.vut.fit.synapse.Synapse;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


/**
 * Network class represent network with single layer of hidden neurons.
 */
public class Network implements Serializable {

    /**
     * Layers.
     */
    private InputNeuron[] inputLayer;
    private List<HiddenNeuron[]> hiddenLayers;
    private OutputNeuron[] outputLayer;

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

    public Network(int inputCount,
                   int[] hiddenCounts,
                   int outputCount,
                   Double learnRate,
                   Double momentum,
                   double[] hiddenLayerBias,
                   Double outputLayerBias
                   ){
        int i;

        globalError = 0.0;

        this.learnRate = learnRate;
        this.momentum = momentum;

        this.inputLayer = new InputNeuron[inputCount];
        for ( i = 0 ; i < inputCount; ++ i ){
            inputLayer[i] = new InputNeuron(0);
        }

        this.hiddenLayers = new ArrayList<>();

        for (int hiddenLayerIndex = 0 ; hiddenLayerIndex < hiddenCounts.length ; ++ hiddenLayerIndex){
            this.hiddenLayers.add(hiddenLayerIndex, new HiddenNeuron[hiddenCounts[hiddenLayerIndex]]);

            for (int hiddenLayerNeuronIndex = 0; hiddenLayerNeuronIndex < hiddenCounts[hiddenLayerIndex]; ++ hiddenLayerNeuronIndex){
                this.hiddenLayers.get(hiddenLayerIndex)[hiddenLayerNeuronIndex] = new HiddenNeuron();
                this.hiddenLayers.get(hiddenLayerIndex)[hiddenLayerNeuronIndex].setBias(hiddenLayerBias[hiddenLayerIndex]);
            }
        }


        this.outputLayer = new OutputNeuron[outputCount];
        for ( i = 0 ; i < outputCount; ++ i ){
            outputLayer[i] = new OutputNeuron();
            outputLayer[i].setBias(outputLayerBias);
        }
    }

    /**
     * Count RootMSE
     * @return RootMSE
     */
    public Double getError(int len){
        Double err = Math.sqrt(globalError / len * outputLayer.length);
        globalError = 0.0; // clear the accumulator
        return err;
    }

    /**
     * Calculate outputs based on input.
     * @param input input of the neural network.
     * @return values of the hidden layer.
     */
    public Double[] calculateOutputs(Double[] input){

        for (int i = 0; i < input.length; ++ i){
            inputLayer[i].setFire(input[i]);
        }

        for (HiddenNeuron[] hiddenLayer:
                hiddenLayers){
            for (HiddenNeuron aHiddenLayer : hiddenLayer) {
                aHiddenLayer.calculateFile();
            }
        }
        int i = 0;
        Double[] result = new Double[outputLayer.length];

        for (OutputNeuron outputNeuron:
             outputLayer) {
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
        for (int i = 0 ; i < outputLayer.length; ++ i){

            globalError += (ideal[i] - outputLayer[i].getFire()) * (ideal[i] - outputLayer[i].getFire());

            Double outNeuronDerivation = outputLayer[i].derivation();

            outputLayer[i].setSigma((ideal[i] - outputLayer[i].getFire()) * outNeuronDerivation );
        }

        /*
        Hidden layers calculation
         */
        for (int layerIndex = hiddenLayers.size() - 1; layerIndex >= 0; -- layerIndex) {

                for (HiddenNeuron hiddenNeuron:
                 hiddenLayers.get(layerIndex)) {
                Double hiddenNeuronDerivation = hiddenNeuron.derivation();

                Double sum = hiddenNeuron.getOutputSynapses().stream().mapToDouble(synapse -> synapse.getWeight() * synapse.getTo().getSigma()).sum();

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
                    Double deltaW = learnRate * outSynapse.getGrad() + momentum * outSynapse.getOldDeltaWeight();
                    outSynapse.adjustWeight(deltaW);
                }
            }
        }

        /*
        Input layer calculation.
         */

        for (InputNeuron inputNeuron:
             inputLayer) {
            for (Synapse synapse:
                 inputNeuron.getOutputSynapses()) {
                synapse.setGrad(inputNeuron.getFire() * synapse.getTo().getSigma());
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
     */
    public void reset(){
        /*
        Connect input neurons and hidden neurons.
         */

        connectLayers(inputLayer, hiddenLayers.get(0));  // Connect input with first hidden

        for (int hiddenLayerIndex = 1; hiddenLayerIndex < hiddenLayers.size() - 1; ++ hiddenLayerIndex){
            connectLayers(hiddenLayers.get(hiddenLayerIndex -1), hiddenLayers.get(hiddenLayerIndex));
        }

        connectLayers(hiddenLayers.get(hiddenLayers.size() - 1), outputLayer);  // Connect last hidden with output

    }

}
