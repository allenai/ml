package org.allenai.ml.optimize;

import org.allenai.ml.linalg.DenseVector;
import org.allenai.ml.linalg.Vector;
import lombok.RequiredArgsConstructor;

public class TestUtils {

    @RequiredArgsConstructor
    public static class MinExample {
        public final GradientFn fn;
        public final Vector argMin;
        public final double min;
    }

    // f(x) = x^2
    public static final MinExample xSquared = new MinExample(
        GradientFn.from(1, x -> GradientFn.Result.of(
            x.at(0) * x.at(0),
            DenseVector.of(2.0 * x.at(0)))),
        DenseVector.of(0.0),
        0.0);

    // f(x,y) = (x-1)^4 + (y + 2.0)^4
    // minimum at (1,-2) with value 0.0
    public static final MinExample quartic = new MinExample(
        GradientFn.from(2, x -> {
            double val = Math.pow(x.at(0) - 1.0, 4.0) + Math.pow(x.at(1) + 2.0, 4.0);
            Vector grad = DenseVector.of(4 * Math.pow(x.at(0) - 1.0, 3.0), 4 * Math.pow(x.at(1) + 2.0, 3.0));
            return GradientFn.Result.of(val, grad);
        }),
        DenseVector.of(1.0, -2.0),
        0.0);

    public static final MinExample approximateCircle = new MinExample(
        new ApproximateGradientFn(4, 1.e-10, (Vector x) -> {
            double x0 = x.at(0);
            double y0 = x.at(1);
            double x1 = x.at(2);
            double y1 = x.at(3);
            double diff =  Math.sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)) - 3;
            return diff * diff;
        }),
        null,
        0.0);
}
