package org.allenai.ml.objective;

import org.allenai.ml.linalg.DenseVector;
import org.allenai.ml.linalg.Vector;
import org.allenai.ml.optimize.GradientFn;
import org.allenai.ml.optimize.NewtonMethod;
import org.allenai.ml.optimize.QuasiNewton;
import lombok.AllArgsConstructor;
import lombok.val;
import org.allenai.ml.util.Parallel;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

import static org.testng.Assert.*;

@Test
public class BatchObjectiveFnTest {

    @AllArgsConstructor
    class RegressionExample {
        double target;
        Vector featVec;
    }


    public void testLinearRegression() {
        List<RegressionExample> examples = Arrays.asList(
            new RegressionExample(1.0, DenseVector.of(0.5, 0.5)),
            new RegressionExample(2.0, DenseVector.of(1.0, 1.0)),
            new RegressionExample(3.0, DenseVector.of(1.5, 1.5)));
        ExampleObjectiveFn<RegressionExample> regressionObjective = (example, weights, grad) -> {
            double guess = example.featVec.dotProduct(weights);
            double diff = guess - example.target;
            grad.addInPlace(diff, example.featVec);
            grad.scaleInPlace(-1.0);
            return -0.5 * diff * diff;
        };
        GradientFn objFn = new BatchObjectiveFn<>(examples, regressionObjective, 2, Parallel.MROpts.withThreads(2));
        val res = objFn.apply(DenseVector.of(2));
        assertEquals(res.fx, 0.5 * (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0), 1.0e-4);

        val minimizer = new NewtonMethod(__ -> QuasiNewton.lbfgs(3));
        val minResult = minimizer.minimize(objFn);
        assertEquals(minResult.fxmin, 0.0, 1.0e-4);
        assertTrue(minResult.xmin.closeTo(DenseVector.of(1.0, 1.0)));
    }
}