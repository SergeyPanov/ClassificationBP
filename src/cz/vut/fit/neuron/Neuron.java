package cz.vut.fit.neuron;

import cz.vut.fit.synapse.Synapse;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Simulate neuron's behavior.
 */
public class Neuron  implements Serializable{
    /**
     * The value neuron fires.
     */
    private Double fire;

    /**
     * Array of input synapses.
     */
    private ArrayList<Synapse> inputSynapses;

    /**
     * Array of output synapses.
     */
    private ArrayList<Synapse> outputSynapses;

    /**
     * Delta value for synapse(needs for learn process).
     */
    private Double delta;

    /**
     * The BIAS value.
     */
    private Double bias;

    /**
     * Activation function.
     * @param sum   Sum of multiplied fires from the preceding neurons and weights.
     * @return  Sigmoid of the 'sum' value.
     */
    private Double sigmoid(Double sum) {
        return 1.0 / (1 + Math.exp(-1.0 * sum));
    }

    /**
     * Derivation of sigmoid.
     * @return
     */
    public Double derivation(){
        return (1 - fire) * fire;
    }

    /**
     * Init neuron.
     */
    Neuron(){
        inputSynapses = new ArrayList<>();
        outputSynapses = new ArrayList<>();
        fire = 0.5 - (Math.random());
    }



    public ArrayList<Synapse> getOutputSynapses(){
        return outputSynapses;
    }

    /**
     * Init neuron with "fire" value.
     * @param fire value should be fired by neuron.
     */
    Neuron(Double fire){
        this();
        this.fire = fire;
    }

    /**
     * Calculate value will be fired.
     */
    public void calculateFile(){
        fire = sigmoid(inputSynapses.stream().mapToDouble(s -> s.getWeight() * s.getFrom().getFire()).sum() + bias);
    }
    public void addInputSynapse(Synapse input){
        inputSynapses.add(input);
    }
    public void addOutputSynapse(Synapse output){
        outputSynapses.add(output);
    }

    public Double getFire() {
        return fire;
    }

    public void setFire(Double fire) {
        this.fire = fire;
    }

    public Double getDelta() {
        return delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }

    public void setBias(Double bias) {
        this.bias = bias;
    }
}
