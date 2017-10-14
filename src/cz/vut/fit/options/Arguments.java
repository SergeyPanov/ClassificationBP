package cz.vut.fit.options;

import org.apache.commons.cli.*;

/**
 * Parsing of input parameters.
 */

public class Arguments {
    /**
     * Parameters holder.
     */
    private CommandLine commandLine;

    private Options options;
    public Arguments(String[] args) throws ParseException {
        options = new Options();

        Option momentum = Option.builder()
                .longOpt("momentum")
                .desc("The momentum value.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option learningRate = Option.builder()
                .longOpt("learning-rate")
                .desc("The learning rate value.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option hiddenLayerBias = Option.builder()
                .longOpt("hidden-bias")
                .desc("The BIAS value for hidden neurons.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option outputLayerBias = Option.builder()
                .longOpt("output-bias")
                .desc("The BIAS value for output neurons.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option inputNeurons = Option.builder()
                .longOpt("input-neurons")
                .desc("Number of input neurons.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option hiddenNeurons = Option.builder()
                .longOpt("hidden-neurons")
                .desc("Number of hidden neurons.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option outputNeurons = Option.builder()
                .longOpt("output-neurons")
                .desc("Number of output neurons.")
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
                .desc("Path to file with vectors need being classified.")
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
                .desc("Desired threshold for RootMSE.")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option iterationsCount = Option.builder()
                .longOpt("iterations-number")
                .desc("Number of iterations.")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .build();

        Option serializeTo = Option.builder()
                .longOpt("serialize-to")
                .desc("Path to the file the network will be serialized.")
                .required(false)
                .type(String.class)
                .hasArg()
                .build();

        Option help = Option.builder()
                .longOpt("help")
                .desc("Print this message.")
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
