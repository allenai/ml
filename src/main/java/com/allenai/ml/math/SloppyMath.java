package com.allenai.ml.math;

import java.util.stream.DoubleStream;

public class SloppyMath {

    private final static double EXP_THRESH = -30.0;

    /**
     * @return log(sum_i exp(vals_i))
     */
    public static double logSumExp(double[] vals) {
        double max = DoubleStream.of(vals).max().orElse(Double.NEGATIVE_INFINITY);
        double sumNegativeDifferences = DoubleStream.of(vals)
                .map(x -> x - max)
                .filter(x -> x > EXP_THRESH)
                .map(Math::exp)
                .sum();
        return sumNegativeDifferences > 0.0
                ? max + Math.log(sumNegativeDifferences)
                : max;
    }
}
