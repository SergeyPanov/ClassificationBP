package cz.vut.fit.reader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class reads input files which contains training set and ideal outputs.
 */
public class InputReader {
    private String path;
    /**
     * Read input file.
     * @return array of doubles.
     */
    public List<List<Double>> getContent() throws IOException {
        List<List<Double>> inputMatrix = new ArrayList<>();
        List<String[]> lines;
        try {
            lines = Files.lines(Paths.get(path)).map(line -> line.split(" +")).collect(Collectors.toList());
        } catch (IOException e) {
            throw new IOException("No such file exists: " + ((NoSuchFileException) e).getFile());
        }catch (NullPointerException e){
            throw new NullPointerException("The path to the file was wrongly set.");
        }

        lines.forEach(splittedLine -> {
                    List<Double> aux = new ArrayList<>();
                    Arrays.asList(splittedLine).forEach(part -> aux.add(Double.valueOf(part)));
                    inputMatrix.add(aux);
                }
        );
        return inputMatrix;
    }

    public void setPath(String path) {
        this.path = path;
    }
}
