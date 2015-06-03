package org.allenai.ml.math;

import java.util.stream.DoubleStream;

public class SloppyMath {

    private final static double EXP_THRESH = -30.0;

    public static double sloppyExp(double x) {
        // assume exp is zero for sufficiently small input
        return x > EXP_THRESH ? Math.exp(x) : 0.0;
    }

    /**
     * __BOTTLENECK__: No higher-level constructs here since this tends to be a perf-bottleneck
     * @return log(sum_i exp(vals_i))
     */
    public static double logSumExp(double[] vals) {
        double max = Double.NEGATIVE_INFINITY;
        for (double val : vals) {
            if (val > max) {
                max = val;
            }
        }
        double sumNegativeDifferences = 0.0;
        for (double val : vals) {
            sumNegativeDifferences += sloppyExp(val-max);
        }
        return sumNegativeDifferences > 0.0
                ? max + Math.log(sumNegativeDifferences)
                : max;
    }
}
