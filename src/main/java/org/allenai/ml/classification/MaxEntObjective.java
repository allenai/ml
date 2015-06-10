package org.allenai.ml.classification;

import com.gs.collections.api.tuple.primitive.IntObjectPair;
import lombok.RequiredArgsConstructor;
import org.allenai.ml.linalg.Vector;
import org.allenai.ml.math.SloppyMath;
import org.allenai.ml.objective.ExampleObjectiveFn;

@RequiredArgsConstructor
public class MaxEntObjective implements ExampleObjectiveFn<IntObjectPair<Vector>> {

    private final int numClasses;

    public static int weightIdx(int predIdx, int classIdx, int numClasses) {
        return predIdx * numClasses + classIdx;
    }

    @Override
    public double evaluate(IntObjectPair<Vector> labeledExample, Vector inParams, Vector outGrad) {
        int trueClassIdx = labeledExample.getOne();
        double[] classProbs = classProbs(labeledExample.getTwo(), inParams, numClasses);
        labeledExample.getTwo().nonZeroEntries().forEach(e -> {
            int predIdx = (int) e.index;
            double predVal = e.value;
            // increment gradient for true class features
            outGrad.inc(weightIdx(predIdx, trueClassIdx, numClasses), 1.0);
            // decrement gradient for all classes by posterior prob
            for (int classIdx = 0; classIdx < numClasses; classIdx++) {
                outGrad.inc(weightIdx(predIdx, classIdx, numClasses), -classProbs[classIdx]);
            }
        });
        return Math.log(classProbs[trueClassIdx]);
    }

    public static double[] classProbs(Vector featVec, Vector weights, int numClasses) {
        double[] logScores = new double[numClasses];
        Vector.Iterator it = featVec.iterator();
        while (!it.isExhausted()) {
            int predIdx = (int) it.index();
            double predVal = it.value();
            for (int classIdx = 0; classIdx < numClasses; classIdx++) {
                int weightIdx = weightIdx(predIdx, classIdx, numClasses);
                logScores[classIdx] +=  weights.at(weightIdx) * predVal;
            }
            it.advance();
        }
        // Exponentiate in place to get probabilities
        double logZ = SloppyMath.logSumExp(logScores);
        for (int classIdx = 0; classIdx < numClasses; classIdx++) {
            logScores[classIdx] = SloppyMath.sloppyExp(logScores[classIdx] - logZ);
        }
        return logScores;
    }
}
