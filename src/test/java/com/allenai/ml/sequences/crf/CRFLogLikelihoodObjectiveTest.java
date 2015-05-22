package com.allenai.ml.sequences.crf;

import com.allenai.ml.linalg.DenseVector;
import com.allenai.ml.linalg.Vector;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static org.testng.Assert.*;

public class CRFLogLikelihoodObjectiveTest {

    CRFWeightsEncoder<String> weightsEncoder = new CRFWeightsEncoder<>(CRFTestUtils.toyStateSpace(), 4, 4);
    CRFLogLikelihoodObjective<String> obj = new CRFLogLikelihoodObjective<>(weightsEncoder);
    Vector params = DenseVector.of(weightsEncoder.numParameters());
    Vector grad;

    @BeforeClass
    public void setUp() {
        grad = DenseVector.of(weightsEncoder.numParameters());
    }

    @Test
    public void testLogLikelihoodObjective() throws Exception {
        CRFIndexedExample ex = CRFTestUtils.toyLabeledExample(new int[]{0, 2, 1});
        double val = obj.evaluate(ex, params, grad);
        // val should be Math.log(1/2)
        assertEquals( val, -Math.log(2), 0.001 );
        int transitionIdx = weightsEncoder.stateSpace.transitionFor(0, 2).get().selfIndex;
        // This '3' comes from the active edge predicate
        int edgeWeightIdx = weightsEncoder.edgeWeightIndex(3, transitionIdx);
        assertTrue(grad.at(edgeWeightIdx) > 0.0);
    }

    @Test(expectedExceptions = Throwable.class)
    public void testThrowsOnUnlabeled() {
        // if the example isn't labeled, should throw
        obj.evaluate(CRFTestUtils.toyExample(), params, grad);
    }

    @Test(expectedExceptions = Throwable.class)
    public void testThrowsOnImpossibleTransition() {
        // this is an impossible gold sequence and should throw
        obj.evaluate(CRFTestUtils.toyLabeledExample(new int[]{1, 0, 2}), params, grad);
    }
}