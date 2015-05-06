package com.allenai.ml.sequences;

import com.gs.collections.impl.factory.Lists;
import com.gs.collections.impl.factory.Sets;
import org.testng.annotations.Test;

import java.util.Arrays;

import static com.gs.collections.impl.tuple.Tuples.pair;
import static org.testng.Assert.*;

@Test
public class ForwardBackwardsTest {

    // toy NP segmentation problem
    String START = "<s>";
    String STOP = "</s>";
    String BEGIN = "BEGIN";
    String MIDDLE = "MIDDLE";
    String OUTSIDE = "OUTSIDE";

    StateSpace<String> segmentationStateSpace = new StateSpace<>(
            Lists.mutable.of(START, STOP, BEGIN, MIDDLE, OUTSIDE),
            Sets.mutable.of(pair(START, BEGIN),
                    pair(START, OUTSIDE),
                    pair(BEGIN, MIDDLE),
                    pair(BEGIN, OUTSIDE),
                    pair(MIDDLE, MIDDLE),
                    pair(MIDDLE, BEGIN),
                    pair(MIDDLE, OUTSIDE),
                    pair(MIDDLE, STOP),
                    pair(OUTSIDE, OUTSIDE),
                    pair(OUTSIDE, BEGIN),
                    pair(OUTSIDE, STOP)));

    public void testSegmentation() throws Exception {
        // Example sentence
        // <s> the_B dog_M hit_O the_B ball_M </s>
        // for this example, BEGIN is the start of a chunk
        // and START is reserved for the start of the entire sequence
        int seqLen = 7;
        double[][] potentials = new double[seqLen-1][segmentationStateSpace.transitions().size()];
        for (double[] row : potentials) {
            Arrays.fill(row, Double.NEGATIVE_INFINITY);
        }
        int startToBeginIdx = segmentationStateSpace.transitionFor(START, BEGIN).get().selfIndex;
        int beginToMiddleIdx = segmentationStateSpace.transitionFor(BEGIN, MIDDLE).get().selfIndex;
        int middleToOutsideIdx = segmentationStateSpace.transitionFor(MIDDLE, OUTSIDE).get().selfIndex;
        int outsideToBeginIdx = segmentationStateSpace.transitionFor(OUTSIDE, BEGIN).get().selfIndex;
        int middleToStopIdx = segmentationStateSpace.transitionFor(MIDDLE, STOP).get().selfIndex;
        potentials[0][startToBeginIdx] = 0.0;
        potentials[1][beginToMiddleIdx] = 0.0;
        potentials[2][middleToOutsideIdx] = 0.0;
        potentials[3][outsideToBeginIdx] = 0.0;
        potentials[4][beginToMiddleIdx] = 0.0;
        potentials[5][middleToStopIdx] = 0.0;
        ForwardBackwards<String> fb = new ForwardBackwards<>(segmentationStateSpace);
        ForwardBackwards<String>.Result result = fb.compute(potentials);
        assertEquals(result.getViterbi(), Arrays.asList(BEGIN, MIDDLE, OUTSIDE, BEGIN, MIDDLE));
        assertEquals(result.getLogZ(), 0.0);
    }
}