## Description
Neural network with Back propagation learning algorithm.

## Arguments
Neural network with Back propagation learning algorithm.
List of available parameters:
    
    --help                       Print this message.
    
    --hidden-bias <arg>          The BIAS value for hidden neurons.
                                 Default is 0.7.
    --hidden-neurons <arg>       Number of hidden neurons.
    --ideal-set <arg>            Path to file with expected results.
    --input-neurons <arg>        Number of input neurons.
    --input-set <arg>            Path to file with vectors need being
                                 classified.
    --iterations-number <arg>    Number of iterations.
    --learning-rate <arg>        The learning rate value. Default is 0.7.
    --momentum <arg>             The momentum value. Default is 0.3.
    --output-bias <arg>          The BIAS value for output neurons.
                                 Default is 0.7.
    --output-neurons <arg>       Number of output neurons.
    --root-mse-threshold <arg>   Desired threshold for RootMSE.
    --serialize-to <arg>         Path to the file the network will be
                                 serialized.
    --trained-network <arg>      Path to the trained network.
    --training-set <arg>         Path to file with training set.
    
## Usage
     `ant build` building project.
     
     `ant clean` remove folders with compiled classes, result jars and serialized networks.
     
     `ant package` create runnable jar file.
    
## Build in example
 
     `ant help` print message about building, packaging, cleaning ant example usage.
 
     `ant train-wine` training network for wine classification.
 
     `ant classify-wine-1` classify first set of unknown wines.
 
     `ant classify-wine-2` classify second set of unknown wines.
 
     `ant classify-wine-3` classify third set of unknown wines.
 

