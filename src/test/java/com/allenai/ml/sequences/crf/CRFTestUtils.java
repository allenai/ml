package com.allenai.ml.sequences.crf;

import com.allenai.ml.linalg.SparseVector;
import com.allenai.ml.linalg.Vector;
import com.allenai.ml.sequences.StateSpace;
import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.tuple.Pair;
import com.gs.collections.impl.factory.Lists;
import com.gs.collections.impl.factory.Sets;
import com.gs.collections.impl.map.mutable.primitive.ObjectDoubleHashMap;
import com.gs.collections.impl.tuple.Tuples;
import lombok.val;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CRFTestUtils {

    static  Vector vec(Object...args) {
        val vec = SparseVector.make(10);
        for (int idx=0; idx < args.length; idx += 2) {
            vec.set(((Integer)args[idx]).longValue(), (double) args[idx+1]);
        }
        return vec;
    }

    static List<Vector> toyNodePreds() {
        return Arrays.asList(
            vec(1, 1.0, 2, 1.0),
            vec(2, 1.0, 3, 1.0, 4, 1.0),
            vec(3, 1.0, 4, 1.0));
    }

    static List<Vector> toyEdgePreds() {
        return Arrays.asList(
            vec(3, 1.0, 4, 1.0),
            vec(2, 1.0, 5, 1.0));
    }

    static <T> ObjectDoubleMap<T> make(Object... args) {
        val m = new ObjectDoubleHashMap<T>();
        for (int idx = 0; idx < args.length; idx += 2) {
            m.put((T) args[idx], (Double) args[idx+1]);
        }
        return m;
    }

    static CRFIndexedExample toyExample() {
        return new CRFIndexedExample(toyNodePreds(), toyEdgePreds());
    }

    static CRFIndexedExample toyLabeledExample(int[] goldLabels) {
        return new CRFIndexedExample(toyNodePreds(), toyEdgePreds(), goldLabels);
    }

    /**
     * State space for regular expression `a*b*`
     * @return
     */
    static StateSpace<String> toyStateSpace() {
        val states = Arrays.asList("<s>","</s>", "a", "b");
        List<Pair<String, String>> transitions = Arrays.asList(
            Tuples.pair("<s>", "a"),
            Tuples.pair("<s>", "b"),
            Tuples.pair("a", "a"),
            Tuples.pair("a", "b"),
            Tuples.pair("a", "</s>"),
            Tuples.pair("b", "</s>"));
        return new StateSpace<String>(states, transitions);
    }
}
