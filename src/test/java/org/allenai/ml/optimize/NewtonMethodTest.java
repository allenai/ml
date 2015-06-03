package org.allenai.ml.optimize;

import org.allenai.ml.linalg.DenseVector;
import lombok.val;
import org.testng.annotations.Test;

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

    public void testMinimizer(GradientFnMinimizer minimizer) {
        testExample(minimizer, TestUtils.quartic);
        testExample(minimizer, TestUtils.xSquared);
    }

    public void testExample(GradientFnMinimizer minimizer, TestUtils.MinExample example) {
        val res = minimizer.minimize(example.fn, DenseVector.of(example.fn.dimension()));
        assertTrue(res.xmin.closeTo(example.argMin, 0.1));
        assertEquals(res.fxmin, example.min, 0.001);
    }
}