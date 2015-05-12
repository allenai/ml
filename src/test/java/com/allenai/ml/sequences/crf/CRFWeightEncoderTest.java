package com.allenai.ml.sequences.crf;

import com.allenai.ml.linalg.DenseVector;
import com.allenai.ml.linalg.Vector;
import com.allenai.ml.sequences.ForwardBackwards;
import com.allenai.ml.sequences.StateSpace;
import com.allenai.ml.sequences.Transition;
import lombok.val;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

import static org.testng.Assert.*;

@Test
public class CRFWeightEncoderTest {

    public void testRowFiller() throws Exception {
        // 2 classes, 2 predicates and 2 offset weights
        Vector weights = DenseVector.of(0.0, 0.0, 1.0, 2.0, 3.0, 4.0);
        Vector featVec = CRFTestUtils.vec(0, 1.0, 1, 2.0);
        double[] rowPotentials = CRFWeightsEncoder.fillRowPotentials(weights, featVec.iterator(), 2, 2);
        // rowPotentials[0] = 1.0 * 1.0 + 3.0 * 2.0 = 7.0
        // rowPotentials[1] = 1.0 * 3.0 + 2.0 * 4.0 = 10.0
        assertEquals(rowPotentials, new double[]{7.0, 10.0});
    }

    public void testEndToEndPotentials() throws Exception {
        StateSpace<String> stateSpace = CRFTestUtils.toyStateSpace();
        CRFIndexedExample toyExample = CRFTestUtils.toyExample();
        val weightEncoder = new CRFWeightsEncoder<String>(stateSpace, 10, 10);
        double[] weights = new double[weightEncoder.numParameters()];
        // Test we can spike potnetials for <s> -> a
        Transition startA = stateSpace.transitionFor("<s>", "a").get();
        // Spike weights for transitions starting with a
        for (int predIdx = 0; predIdx < weightEncoder.numEdgePredicates; ++predIdx) {
            int edgeWeightIndex = weightEncoder.edgeWeightIndex(predIdx, startA.selfIndex);
            weights[edgeWeightIndex] = 1.0;
        }

        double[][] potentials = weightEncoder.fillPotentials(DenseVector.of(weights), toyExample);
        val fbResult = new ForwardBackwards<String>(stateSpace).compute(potentials);
        List<String> aBiasedViterbi = fbResult.getViterbi();
        assertEquals(aBiasedViterbi, Arrays.asList("a"));

        Arrays.fill(weights, 0.0);
        // Test we can spike potnetials for <s> -> a
        Transition startB = stateSpace.transitionFor("<s>", "b").get();
        // Spike weights for transitions starting with a
        for (int predIdx = 0; predIdx < weightEncoder.numEdgePredicates; ++predIdx) {
            int edgeWeightIndex = weightEncoder.edgeWeightIndex(predIdx, startB.selfIndex);
            weights[edgeWeightIndex] = 1.0;
        }
        potentials = weightEncoder.fillPotentials(DenseVector.of(weights), toyExample);
        List<String> bBiasedViterbi = new ForwardBackwards<String>(stateSpace)
            .compute(potentials)
            .getViterbi();
        assertEquals(bBiasedViterbi, Arrays.asList("b"));
    }
}