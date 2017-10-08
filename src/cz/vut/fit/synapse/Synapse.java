package cz.vut.fit.synapse;

import cz.vut.fit.neuron.Neuron;

public class Synapse {
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
    private double weight;

    private double oldDeltaWeight;

    private double grad;



    public Synapse(){
        weight = 0.5 - (Math.random());
        oldDeltaWeight = 0.0;
    }

    /**
     * Adjust weight base on the delta.
     * @param delta step from old weight to new weight.
     */
    public void adjustWeight(double delta){
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

    public double getWeight() {
        return weight;
    }



    public double getGrad() {
        return grad;
    }

    public void setGrad(double grad) {
        this.grad = grad;
    }

    public double getOldDeltaWeight() {
        return oldDeltaWeight;
    }
}
