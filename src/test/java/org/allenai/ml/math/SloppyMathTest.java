package org.allenai.ml.math;

import static org.testng.Assert.*;
import org.testng.annotations.Test;

import java.util.Random;
import java.util.stream.DoubleStream;

@Test
public class SloppyMathTest {
    public void testLogSumExp() throws Exception {
        Random rand = new Random(0L);
        int MIN = -100;
        int MAX = 100;
        int N = 100;
        int numRandCases = 100;
        for (int idx=0; idx < numRandCases; ++idx) {
            double[] vals = DoubleStream
                .generate(() -> rand.nextInt(MAX-MIN) + MIN)
                .limit(N)
                .toArray();
            double fast = SloppyMath.logSumExp(vals);
            double slow = Math.log(DoubleStream.of(vals).map(Math::exp).sum());
            assertEquals(fast, slow, 1.0e-10);
        }
   }
}