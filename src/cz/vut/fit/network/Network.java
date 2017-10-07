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

    /**
     * Changes in the errors.
     */
    private double errorDelta[];


    public Network(int inputCount,
                   int hiddenCount,
                   int outputCount,
                   double learnRate,
                   double momentum){
        int i;

        this.learnRate = learnRate;
        this.momentum = momentum;


        this.totalNeurons = inputCount + hiddenCount + outputCount;

        this.thresholds = new double[this.totalNeurons];
        this.thresholdDelta = new double[this.totalNeurons];
        this.accThresholdDelta = new double[this.totalNeurons];
        
        this.totalSynapses = inputCount * hiddenCount + hiddenCount * outputCount;
        this.synapseDelta = new double[totalSynapses];
        this.accSynapseDelta = new double[totalSynapses];


        this.error = new double[totalNeurons];
        this.errorDelta = new double[totalNeurons];

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

    public double getError(int len){
        double err = Math.sqrt(globalError / (len * outputCount));
        globalError = 0; // clear the accumulator
        return err;
    }

    public double[] calculateOutputs(double input[]){

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


    public void learn(double ideal[]){
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
        return;
    }

}
