package org.allenai.ml.eval;

import com.gs.collections.api.tuple.primitive.IntIntPair;
import com.gs.collections.impl.tuple.primitive.PrimitiveTuples;

public class Accuracy {
    private int numCorrect;
    private int total;

    public void update(boolean correct) {
        if (correct) {
            numCorrect++;
        }
        total++;
    }

    public void combine(Accuracy other) {
        numCorrect += other.numCorrect;
        total += other.total;
    }

    public double accuracy() {
        return total > 0 ? (double)numCorrect/(double)total : 0.0;
    }

    public IntIntPair stats() {
        return PrimitiveTuples.pair(numCorrect, total);
    }
}
