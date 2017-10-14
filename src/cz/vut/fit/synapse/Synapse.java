package cz.vut.fit.synapse;

import cz.vut.fit.neuron.Neuron;

import java.io.Serializable;

/**
 * Synapse connect two neurons.
 */
public class Synapse implements Serializable {
    /**
     * Left neuron.
     */
    private Neuron from;

    /**
     * Right neuron.
     */
    private Neuron to;

    /**
     * Weight of the synapse.
     */
    private Double weight;

    /**
     * Old delta value(need for learning).
     */
    private Double oldDeltaWeight;

    /**
     * Gradient value(needs for learning).
     */
    private Double grad;



    public Synapse(){
        weight = 0.5 - (Math.random());
        oldDeltaWeight = 0.0;
    }

    /**
     * Adjust weight base on the delta.
     * @param delta step from old weight to new weight.
     */
    public void adjustWeight(Double delta){
        oldDeltaWeight = delta;
        weight += delta;
    }

    public Neuron getFrom() {
        return from;
    }

    public void setFrom(Neuron from) {
        this.from = from;
    }

    public Neuron getTo() {
        return to;
    }

    public void setTo(Neuron to) {
        this.to = to;
    }

    public Double getWeight() {
        return weight;
    }



    public Double getGrad() {
        return grad;
    }

    public void setGrad(Double grad) {
        this.grad = grad;
    }

    public Double getOldDeltaWeight() {
        return oldDeltaWeight;
    }
}
