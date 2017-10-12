package cz.vut.fit.neuron;

import cz.vut.fit.synapse.Synapse;

import java.io.Serializable;
import java.util.ArrayList;

public class Neuron  implements Serializable{
    /**
     * Output of the neuron.
     */
    private Double fire;

    private ArrayList<Synapse> inputSynapses;
    private ArrayList<Synapse> outputSynapses;

    private double sigma;

    private double bias;


    private Double sigmoid(Double sum) {
        return 1.0 / (1 + Math.exp(-1.0 * sum));
    }

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
    Neuron(double fire){
        this();
        this.fire = fire;
    }

    public void calculateFile(){
        fire = sigmoid(inputSynapses.stream().mapToDouble(s -> s.getWeight() * s.getFrom().getFire()).sum() + bias);
    }

    public void addInputSynapse(Synapse input){
        inputSynapses.add(input);
    }
    public void addOutputSynapse(Synapse output){
        outputSynapses.add(output);
    }

    public double getFire() {
        return fire;
    }

    public void setFire(double fire) {
        this.fire = fire;
    }

    public double getSigma() {
        return sigma;
    }

    public void setSigma(double sigma) {
        this.sigma = sigma;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
