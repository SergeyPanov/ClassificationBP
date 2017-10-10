package cz.vut.fit.main;

import cz.vut.fit.network.Network;
import cz.vut.fit.options.Arguments;
import cz.vut.fit.reader.InputReader;
import cz.vut.fit.stopcondition.StopCondition;

import java.io.*;
import java.util.List;

public class Main {

    private static Network deserializeNetwork() throws IOException, ClassNotFoundException {
        Network deserializedNetwork;
        ObjectInputStream objectInputStream;
        objectInputStream = new ObjectInputStream(new FileInputStream(arguments.getCommandLine().getOptionValue("trained-network")));

        deserializedNetwork = (Network) objectInputStream.readObject();
        objectInputStream.close();
        return deserializedNetwork;
    }


    private static void serializeNetwork(Network network) throws IOException {
        File file = new File(arguments.getCommandLine().getOptionValue("training-set"));

        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(arguments.getCommandLine().getOptionValue("serialize-to", file.getName() + ".nnet")));
        out.writeObject(network);
        out.close();

    }

    private static Arguments arguments;

    /**
     * Method is called for learning procedure.
     */
    private static void learn() throws Exception {

        if (
                            arguments.getCommandLine().getOptionValue("input-neurons") == null ||
                            Integer.valueOf(arguments.getCommandLine().getOptionValue("input-neurons")) <= 0||
                            arguments.getCommandLine().getOptionValue("hidden-neurons") == null ||
                            Integer.valueOf(arguments.getCommandLine().getOptionValue("hidden-neurons")) <= 0 ||
                            arguments.getCommandLine().getOptionValue("output-neurons") == null ||
                            Integer.valueOf(arguments.getCommandLine().getOptionValue("output-neurons")) <= 0
                ){
            throw new Exception("Parameters 'input-neurons', 'hidden-neurons' and output-neurons are required for training network and should be > 0.");
        }


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

        input = inputReader.getContent();


        // Read ideal results
        List<List<Double>> ideal;
        inputReader.setPath(arguments.getCommandLine().getOptionValue("ideal-set"));


        ideal = inputReader.getContent();


        int iterations;

        StopCondition stopCondition = StopCondition.THRESHOLD;
        if (arguments.getCommandLine().getOptionValue("iterations-number") != null){
            stopCondition = StopCondition.ITERATIONS;
        }
        iterations = Integer.valueOf(arguments.getCommandLine().getOptionValue("iterations-number", "-1"));

        double rootMSEthreshold;

        if (arguments.getCommandLine().getOptionValue("root-mse-threshold") != null){
            stopCondition = StopCondition.THRESHOLD;
        }
        rootMSEthreshold = Double.valueOf(arguments.getCommandLine().getOptionValue("root-mse-threshold", "0.3"));


        boolean isFinish = false;
        int i = 0;

        while (!isFinish){

            for (int j = 0 ; j < input.size(); ++ j){
                Double[] inputLine = input.get(j).toArray(new Double[]{});
                network.calculateOutputs(inputLine);

                Double[] idealLine = ideal.get(j).toArray(new Double[]{});
                network.learn(idealLine);
            }
            double rootMSE = network.getError(ideal.size());
            System.out.println("Iteration: #" + i + " Error: " + rootMSE);
            ++i;

            switch (stopCondition){
                case THRESHOLD:
                    if (rootMSE <= rootMSEthreshold) isFinish = true;
                    break;
                case ITERATIONS:
                    if (i >= iterations) isFinish = true;
                    break;
            }
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
