package cz.vut.fit.options;

import org.apache.commons.cli.*;

public class Arguments {

    private CommandLine commandLine;
    private Options options;

    public Arguments(String[] args) throws ParseException {
        options = new Options();

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

        Option hiddenLayerBiases = Option.builder()
                .longOpt("hidden-biases")
                .desc("Hidden layer BIAS value.")
                .required(false)
                .type(Double[].class)
                .hasArgs()
                .build();

        Option outputLayerBias = Option.builder()
                .longOpt("output-bias")
                .desc("Output layer BIAS value.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option outputLayer = Option.builder()
                .longOpt("input-layer")
                .desc("Amount of input neurons.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option hiddenNeurons = Option.builder()
                .longOpt("hidden-layers")
                .desc("Amount of hidden neurons.")
                .required(false)
                .type(Integer[].class)
                .hasArgs()
                .build();

        Option outputNeurons = Option.builder()
                .longOpt("output-layer")
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

        Option help = Option.builder()
                .longOpt("help")
                .build();

        options
                .addOption(momentum)
                .addOption(learningRate)
                .addOption(hiddenLayerBiases)
                .addOption(outputLayerBias)
                .addOption(outputLayer)
                .addOption(hiddenNeurons)
                .addOption(outputNeurons)
                .addOption(trainingSetPath)
                .addOption(idealResultsPath)
                .addOption(inputSetPath)
                .addOption(trainedNetwork)
                .addOption(rootMSEthreshold)
                .addOption(iterationsCount)
                .addOption(serializeTo)
                .addOption(help)
        ;

        CommandLineParser parser = new DefaultParser();

        commandLine = parser.parse(options, args);
    }

    public Options getOptions() {
        return options;
    }

    public CommandLine getCommandLine() {
        return commandLine;
    }
}
