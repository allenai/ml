package org.allenai.ml.classification;


import com.gs.collections.api.map.primitive.ObjectDoubleMap;

@FunctionalInterface
public interface FeatureExtractor<D, F> {
    ObjectDoubleMap<F> features(D datum);
}
