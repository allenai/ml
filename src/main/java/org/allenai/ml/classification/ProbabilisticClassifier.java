package org.allenai.ml.classification;

import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.tuple.primitive.ObjectDoublePair;

public interface ProbabilisticClassifier<D, L> extends Classifier<D, L> {
    ObjectDoubleMap<L> probabilities(D datum);

    /**
     * Default to just the `argMax` of the `probabilties`
     */
    default L bestGuess(D datum) {
        ObjectDoublePair<L> max = null;
        for (ObjectDoublePair<L> pair : probabilities(datum).keyValuesView()) {
            if (max == null || pair.getTwo() > max.getTwo()) {
                max = pair;
            }
        }
        return max.getOne();
    }
}
