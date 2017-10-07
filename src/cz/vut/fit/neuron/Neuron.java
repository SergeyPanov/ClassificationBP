package cz.vut.fit.neuron;

import cz.vut.fit.synapse.Synapse;

import java.util.ArrayList;

public class Neuron {
    /**
     * Output of the neuron.
     */
    private double fire;

    /**
     * List of input synapses.
     */
    private ArrayList<Synapse> connections;

    /**
     * Init neuron.
     */
    Neuron(){
        connections = new ArrayList<>();
        fire = 0.5 - (Math.random());
    }

    /**
     * Init neuron with "fire" value.
     * @param fire
     */
    Neuron(double fire){
        this();
        this.fire = fire;
    }






    public ArrayList<Synapse> getConnections() {
        return connections;
    }

    public void addConnection(Synapse connection) {
        this.connections.add(connection);
    }


    public double getFire() {
        return fire;
    }

    public void setFire(double fire) {
        this.fire = fire;
    }

}
