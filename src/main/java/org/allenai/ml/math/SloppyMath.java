package org.allenai.ml.math;

import com.gs.collections.api.list.primitive.DoubleList;
import com.gs.collections.impl.list.mutable.primitive.DoubleArrayList;

import java.util.stream.DoubleStream;

public class SloppyMath {

    public final static double EXP_THRESH = -30.0;

    public static double sloppyExp(double x) {
        // assume exp is zero for sufficiently small input
        return x > EXP_THRESH ? Math.exp(x) : 0.0;
    }

    public static double logSumExp(double[] vals) {
        return logSumExp(DoubleArrayList.newListWith(vals));
    }

    /**
     * __BOTTLENECK__: No higher-level constructs here since this tends to be a perf-bottleneck
     * @return log(sum_i exp(vals_i))
     */
    public static double logSumExp(DoubleList vals) {
        double max = Double.NEGATIVE_INFINITY;
        int n = vals.size();
        for (int idx = 0; idx < n; idx++) {
            double val = vals.get(idx);
            if (val > max) {
                max = val;
            }
        }
        double sumNegativeDifferences = 0.0;
        for (int idx = 0; idx < n; idx++) {
            double val = vals.get(idx);
            sumNegativeDifferences += sloppyExp(val-max);
        }
        return sumNegativeDifferences > 0.0
                ? max + Math.log(sumNegativeDifferences)
                : max;
    }
}
