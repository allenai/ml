package org.allenai.ml.optimize;

import lombok.RequiredArgsConstructor;
import org.allenai.ml.linalg.DenseVector;
import org.allenai.ml.linalg.Vector;

import java.util.function.ToDoubleFunction;

@RequiredArgsConstructor
public class ApproximateGradientFn implements GradientFn {

    private final long dimension;
    private final double epsilon;
    private final ToDoubleFunction<Vector> fn;

    @Override
    public Result apply(Vector vec) {
        if (vec.dimension() != dimension) {
            throw new IllegalArgumentException("Input doesn't agree on dimension");
        }
        double fx = fn.applyAsDouble(vec);
        Vector grad = DenseVector.of(vec.dimension());
        for (int idx = 0; idx < vec.dimension(); idx++) {
            vec.inc(idx, epsilon);
            double fxPlusDelta = fn.applyAsDouble(vec);
            grad.set(idx, (fxPlusDelta - fx) / epsilon);
            vec.inc(idx, -epsilon);
        }
        return Result.of(fx, grad);
    }

    @Override
    public boolean isGradientApproximate() {
        return true;
    }

    @Override
    public long dimension() {
        return dimension;
    }
}
