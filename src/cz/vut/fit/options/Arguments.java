package cz.vut.fit.options;

import org.apache.commons.cli.*;

public class Arguments {

    private CommandLine commandLine;


    public Arguments(String[] args) throws ParseException {
        Options options = new Options();

        Option momentum = Option.builder()
                .longOpt("momentum")
                .desc("Momentum value.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option learningRate = Option.builder()
                .longOpt("learning-rate")
                .desc("Learning rate value.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option hiddenLayerBias = Option.builder()
                .longOpt("hidden-bias")
                .desc("Hidden layer BIAS value.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option outputLayerBias = Option.builder()
                .longOpt("output-bias")
                .desc("Output layer BIAS value.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option inputNeurons = Option.builder()
                .longOpt("input-neurons")
                .desc("Amount of input neurons.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option hiddenNeurons = Option.builder()
                .longOpt("hidden-neurons")
                .desc("Amount of hidden neurons.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option outputNeurons = Option.builder()
                .longOpt("output-neurons")
                .desc("Amount of output neurons.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option trainingSetPath = Option.builder()
                .longOpt("training-set")
                .desc("Path to file with training set.")
                .type(String.class)
                .hasArg()
                .build();

        Option idealResultsPath = Option.builder()
                .longOpt("ideal-set")
                .desc("Path to file with expected results.")
                .type(String.class)
                .hasArg()
                .build();

        Option inputSetPath = Option.builder()
                .longOpt("input-set")
                .desc("Path to file with vectors needed to be classified.")
                .type(String.class)
                .hasArg()
                .build();

        Option trainedNetwork = Option.builder()
                .longOpt("trained-network")
                .desc("Path to the trained network.")
                .type(String.class)
                .hasArg()
                .required(false)
                .build();

        Option rootMSEthreshold = Option.builder()
                .longOpt("root-mse-threshold")
                .desc("Desired precision set by RootMSE for training.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option iterationsCount = Option.builder()
                .longOpt("iterations-number")
                .desc("Amount of iterations for training.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option serializeTo = Option.builder()
                .longOpt("serialize-to")
                .desc("Path to the file for serialization of trained network")
                .required(false)
                .type(String.class)
                .hasArg()
                .build();

        options
                .addOption(momentum)
                .addOption(learningRate)
                .addOption(hiddenLayerBias)
                .addOption(outputLayerBias)
                .addOption(inputNeurons)
                .addOption(hiddenNeurons)
                .addOption(outputNeurons)
                .addOption(trainingSetPath)
                .addOption(idealResultsPath)
                .addOption(inputSetPath)
                .addOption(trainedNetwork)
                .addOption(rootMSEthreshold)
                .addOption(iterationsCount)
                .addOption(serializeTo)
        ;

        CommandLineParser parser = new DefaultParser();

        commandLine = parser.parse(options, args);
    }


    public CommandLine getCommandLine() {
        return commandLine;
    }
}
