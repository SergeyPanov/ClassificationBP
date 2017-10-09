package cz.vut.fit.main;

import cz.vut.fit.network.Network;
import cz.vut.fit.options.Arguments;
import cz.vut.fit.reader.InputReader;

import java.io.*;
import java.util.List;

public class Main {

    private static Network deserializeNetwork() throws IOException, ClassNotFoundException {
        Network deserializedNetwork;
        ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(arguments.getCommandLine().getOptionValue("trained-network")));

        deserializedNetwork = (Network) objectInputStream.readObject();
        objectInputStream.close();
        return deserializedNetwork;
    }


    private static void serializeNetwork(Network network) throws IOException {
        File file = new File(arguments.getCommandLine().getOptionValue("training-set"));

        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file.getName() + ".nnet"));
        out.writeObject(network);
        out.close();

    }

    private static Arguments arguments;

    /**
     * Method is called for learning procedure.
     */
    private static void learn(){


        Network network = new Network(
                Integer.valueOf(arguments.getCommandLine().getOptionValue("input-neurons")),
                Integer.valueOf(arguments.getCommandLine().getOptionValue("hidden-neurons")),
                Integer.valueOf(arguments.getCommandLine().getOptionValue("output-neurons")),
                Double.valueOf(arguments.getCommandLine().getOptionValue("learning-rate", "0.7")),
                Double.valueOf(arguments.getCommandLine().getOptionValue("momentum", "0.3")),
                Double.valueOf(arguments.getCommandLine().getOptionValue("hidden-bias", "0.7")),
                Double.valueOf(arguments.getCommandLine().getOptionValue("output-bias", "0.7"))
        );


        network.reset();

        // Read input file
        InputReader inputReader = new InputReader();
        inputReader.setPath(arguments.getCommandLine().getOptionValue("training-set"));
        List<List<Double>> input;
        try {
            input = inputReader.getContent();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Read ideal results
        List<List<Double>> ideal;
        inputReader.setPath(arguments.getCommandLine().getOptionValue("ideal-set"));

        try {
            ideal = inputReader.getContent();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        for (int i = 0; i < 10_00; ++ i){
            for (int j = 0 ; j < input.size(); ++ j){
                Double[] inputLine = input.get(j).toArray(new Double[]{});
                network.calculateOutputs(inputLine);

                Double[] idealLine = ideal.get(j).toArray(new Double[]{});
                network.learn(idealLine);
            }
            System.out.println("Iteration: #" + i + " Error: " + network.getError(ideal.size()));
        }

        try {
            serializeNetwork(network);
        } catch (IOException e) {
            System.err.println("The serialization of the network was fell.");
            e.printStackTrace();
        }

    }

    private static void classification() throws IOException, ClassNotFoundException {
        Network network;

        network = deserializeNetwork();

        InputReader inputReader = new InputReader();

        // Read file with set need being classified
        List<List<Double>> inputSet;
        inputReader.setPath(arguments.getCommandLine().getOptionValue("input-set"));

        try {
            inputSet = inputReader.getContent();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        System.out.println("Recall");
        for (List<Double> anInput : inputSet) {
            for (Double anAnInput : anInput) {
                System.out.print(anAnInput + " : ");
            }
            Double[] inputAux = anInput.toArray(new Double[]{});
            Double out[] = network.calculateOutputs(inputAux);
            System.out.print("= ");
            for (double d :
                    out) {
                System.out.print(d + " ");
            }
            System.out.println("");
        }

    }

    public static void main(String[] args) throws Exception {

        arguments = new Arguments(args);

        /*
        If training-set parameter is set the learning method should proceed
         */
        if (arguments.getCommandLine().getOptionValue("training-set") != null){
            learn();
        }else {
            if (arguments.getCommandLine().getOptionValue("input-set") != null){
                classification();
            }
        }
    }
}
