package cz.vut.fit.main;

import cz.vut.fit.network.Network;
import cz.vut.fit.options.Arguments;
import cz.vut.fit.reader.InputReader;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {

        Arguments arguments = new Arguments(args);

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
        inputReader.setPath("G:\\IdeaProjects\\ClassificationBP\\src\\resources\\ZooNormalizedData");
        List<List<Double>> input;
        try {
            input = inputReader.getContent();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        List<List<Double>> ideal;
        inputReader.setPath("G:\\IdeaProjects\\ClassificationBP\\src\\resources\\ZooIdeal");

        try {
            ideal = inputReader.getContent();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        System.out.println(ideal);

        for (int i = 0; i < 10_000; ++ i){
            for (int j = 0 ; j < input.size(); ++ j){
                Double[] inputLine = input.get(j).toArray(new Double[]{});
                network.calculateOutputs(inputLine);

                Double[] idealLine = ideal.get(j).toArray(new Double[]{});
                network.learn(idealLine);
            }
            System.out.println("Iteration: #" + i + " Error: " + network.getError(ideal.size()));
        }
        System.out.println("Recall");
        for (List<Double> anInput : input) {
            for (Double anAnInput : anInput) {
                System.out.print(anAnInput + " : ");
            }
            Double[] inputAux = anInput.toArray(new Double[]{});
            double out[] = network.calculateOutputs(inputAux);
            System.out.print("= ");
            for (double d :
                    out) {
                System.out.print(d + " ");
            }
            System.out.println("");
        }

    }
}
