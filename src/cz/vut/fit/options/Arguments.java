package cz.vut.fit.options;

import org.apache.commons.cli.*;

public class Arguments {

    private CommandLine commandLine;


    public Arguments(String[] args) throws ParseException {
        Options options = new Options();

        Option momentum = Option.builder()
                .longOpt("momentum")
                .desc("Momentum value")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option learningRate = Option.builder()
                .longOpt("learning-rate")
                .desc("Learning rate value")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option hiddenLayerBias = Option.builder()
                .longOpt("hidden-bias")
                .desc("Hidden layer BIAS value")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option outputLayerBias = Option.builder()
                .longOpt("output-bias")
                .desc("Output layer BIAS value")
                .required(false)
                .type(Double.class)
                .hasArg()
                .build();

        Option outputNeurons = Option.builder()
                .longOpt("output-neurons")
                .desc("Amount of output neurons")
                .required(true)
                .type(Integer.class)
                .hasArg()
                .build();

        Option hiddenNeurons = Option.builder()
                .longOpt("hidden-neurons")
                .desc("Amount of hidden neurons")
                .required(true)
                .type(Integer.class)
                .hasArg()
                .build();

        Option inputNeurons = Option.builder()
                .longOpt("input-neurons")
                .desc("Amount of input neurons")
                .required(true)
                .type(Integer.class)
                .hasArg()
                .build();
        options
                .addOption(momentum)
                .addOption(learningRate)
                .addOption(hiddenLayerBias)
                .addOption(outputLayerBias)
                .addOption(inputNeurons)
                .addOption(hiddenNeurons)
                .addOption(outputNeurons);

        CommandLineParser parser = new DefaultParser();

        commandLine = parser.parse(options, args);
    }


    public CommandLine getCommandLine() {
        return commandLine;
    }
}
