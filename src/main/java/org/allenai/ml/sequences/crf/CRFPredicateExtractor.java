package org.allenai.ml.sequences.crf;

import com.gs.collections.api.map.primitive.ObjectDoubleMap;

import java.util.List;

public interface CRFPredicateExtractor<O, F> {
    List<ObjectDoubleMap<F>> nodePredicates(List<O> elems);
    List<ObjectDoubleMap<F>> edgePredicates(List<O> elems);
}
