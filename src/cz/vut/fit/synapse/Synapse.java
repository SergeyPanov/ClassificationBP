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



    public Synapse(){
        weight = 0.5 - (Math.random());
    }

    /**
     * Adjust weight base on the delta.
     * @param delta
     */
    public void adjustWeight(double delta){
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

    public void setWeight(double weight) {
        this.weight = weight;
    }
}
