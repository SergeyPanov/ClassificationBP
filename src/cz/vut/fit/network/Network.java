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

    private InputNeuron[] inputNeurons;
    private HiddenNeuron[] hiddenNeurons;
    private OutputNeuron[] outputNeurons;


    /**
     * Count of input neurons.
     */
    private int inputCount;

    /**
     * Count of hidden neurons.
     */
    private int hiddenCount;

    /**
     * Count of output neurons.
     */
    private int outputCount;

    /**
     * Total count of neurons.
     */
    private int totalNeurons;

    /**
     * The threshold values along with the weights.
     */
    private double[] thresholds;

    /**
     * The threshold deltas.
     */
    private double thresholdDelta[];

    /**
     * The accumulation of the threshold deltas.
     */
    private double accThresholdDelta[];


    /**
     * Accumulates synapse's delta's for training.
     */
    private double accSynapseDelta[];

    /**
     * The changes should be applied to the synapse's weight.
     */
    private double synapseDelta[];


    /**
     * Total synapses in the network.
     */
    private int totalSynapses;

    /**
     * The learning rate.
     */
    private final double learnRate;

    /**
     * The momentum.
     */
    private final double momentum;

    /**
     * Error from the last calculation.
     */
    private double error[];

    /**
     * The global RootMSE.
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

        this.inputCount = inputCount;
        this.hiddenCount = hiddenCount;
        this.outputCount = outputCount;

        this.totalNeurons = inputCount + hiddenCount + outputCount;

        this.thresholds = new double[this.totalNeurons];
        this.thresholdDelta = new double[this.totalNeurons];
        this.accThresholdDelta = new double[this.totalNeurons];
        
        this.totalSynapses = inputCount * hiddenCount + hiddenCount * outputCount;
        this.synapseDelta = new double[totalSynapses];
        this.accSynapseDelta = new double[totalSynapses];


        this.error = new double[totalNeurons];

        this.inputNeurons = new InputNeuron[inputCount];
        for ( i = 0 ; i < inputCount; ++ i ){
            inputNeurons[i] = new InputNeuron(0);
        }

        this.hiddenNeurons = new HiddenNeuron[hiddenCount];
        for ( i = 0;  i < hiddenCount; ++ i){
            hiddenNeurons[i] = new HiddenNeuron();
        }

        this.outputNeurons = new OutputNeuron[outputCount];
        for ( i = 0 ; i < outputCount; ++ i ){
            outputNeurons[i] = new OutputNeuron();
        }
    }

    /**
     * Activation function "sigmoid".
     * @param sum
     * @return
     */
    private double sigmoid(double sum){
        return 1.0 / (1 + Math.exp(-1.0 * sum));
    }


    /**
     * Calculate root mean square error.
     * @param desired
     * @return
     */
    private double calculateError(double desired[]){
        //TODO: Calculate error

        for (int i = 0 ; i < inputNeurons.length; ++ i){
            error[i] = 0;
        }

        /*
        Count layer errors and deltas for output.
         */

//        for ( int i = 0; i < outputNeurons.length; ++ i ){
//            error[outputNeurons.length + i] = desired[]
//        }

        return 0.0;
    }

    /**
     * Calculate output of the network for given input.
     * @param input
     */
    public void calculateOutput(double input[]){
        int i;
        /*
        Set input values
         */
        for ( i = 0; i < inputCount; ++ i ){
            inputNeurons[i].setFire(input[i]);
        }

        /*
        Calculate outputs for hidden layer.
         */
        for ( i = 0 ; i < hiddenNeurons.length; ++ i ){
            double sum = thresholds[inputNeurons.length + i];

            for (Synapse synapse:
                 hiddenNeurons[i].getConnections()) {
                sum += synapse.getFrom().getFire() * synapse.getWeight();
            }
            hiddenNeurons[i].setFire(sigmoid(sum));
        }

        /*
        Calculate outputs for output layers.
         */
        for ( i = 0; i < outputNeurons.length; ++ i ){
            double sum = thresholds[hiddenNeurons.length + 1];

            for ( Synapse synapse:
                    outputNeurons[i].getConnections()){
                sum += synapse.getFrom().getFire() * synapse.getWeight();
            }
            outputNeurons[i].setFire(sigmoid(sum));
        }
    }


    private void connectLayers(Neuron[] first, Neuron[] second){
        for (Neuron aFirst : first) {
            for (Neuron aSecond : second) {

                Synapse synapse = new Synapse(); // Create new connection
                synapse.setFrom(aFirst);    // Set "from" as input neuron
                synapse.setTo(aSecond); // Set "to" as hidden neuron

                aSecond.addConnection(synapse);
            }
        }
    }

    /**
     * Reset the network.
     */
    public void reset(){
        int i;
        /*
          Init thresholds.
         */
        for ( i = 0; i < totalNeurons; ++ i ){
            thresholds[i] = 0.5 - (Math.random());
            thresholdDelta[i] = 0;
            accThresholdDelta[i] = 0;
        }
        connectLayers(inputNeurons, hiddenNeurons);
        connectLayers(hiddenNeurons, outputNeurons);
    }

}
