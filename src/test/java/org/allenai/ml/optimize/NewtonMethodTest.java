package org.allenai.ml.optimize;

import org.allenai.ml.linalg.DenseVector;
import lombok.val;
import org.testng.annotations.Test;

import java.util.Random;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

public class NewtonMethodTest {

    @Test
    public void testGradientDescent() throws Exception {
        val minimizer = new NewtonMethod(__ -> QuasiNewton.gradientDescent());
        testMinimizer(minimizer);
    }

    @Test
    public void testLBFGS() throws Exception {
        testMinimizer(new NewtonMethod(__ -> QuasiNewton.lbfgs(1)));
        testMinimizer(new NewtonMethod(__ -> QuasiNewton.lbfgs(3)));
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testZeroDiffLBFGS() throws Exception {
        QuasiNewton.lbfgs(1).update(DenseVector.of(10), DenseVector.of(10));
    }

    @Test
    public void testMinimizer(GradientFnMinimizer minimizer) {
        testExample(minimizer, TestUtils.quartic);
        testExample(minimizer, TestUtils.xSquared);
    }

    private final static Random rand = new Random(0L);

    @Test
    public void testExample(GradientFnMinimizer minimizer, TestUtils.MinExample example) {
        val initVec = DenseVector.of(example.fn.dimension());
        for (int idx = 0; idx < initVec.dimension(); idx++) {
            double x = 2.0 * rand.nextDouble() - 1.0;
            initVec.set(idx, x);
        }
        val res = minimizer.minimize(example.fn, initVec);
        if (example.argMin != null) {
            assertTrue(res.xmin.closeTo(example.argMin, 0.1));
        }
        assertEquals(res.fxmin, example.min, 0.001);
    }

    @Test
    public void testApproximate() throws Exception {
        testExample(new NewtonMethod(__ -> QuasiNewton.lbfgs(3)), TestUtils.approximateCircle);
    }
}