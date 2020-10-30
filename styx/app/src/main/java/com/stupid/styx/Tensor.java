package com.stupid.styx;

import java.util.Arrays;

public class Tensor {
    int[] shape;
    float[] data;

    @Override
    public String toString() {
        String output = "";
        output += "shape: " + Arrays.toString(shape);
        output += " data: " + Arrays.toString(data);
        return output;
    }
}