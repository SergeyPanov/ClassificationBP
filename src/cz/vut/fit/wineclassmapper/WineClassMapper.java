package cz.vut.fit.wineclassmapper;


import java.util.List;
import java.util.stream.IntStream;

public class WineClassMapper {


    public static String getClass(List<Double> vector){

        switch (IntStream.range(0, vector.size())
                .reduce((i,j) -> vector.get(i) > vector.get(j) ? i : j)
                .orElse(-1)

                ){
            case 0: return "A";
            case 1: return "B";
            case 2: return "C";
            default: return "Unknown";
        }

    }



}
