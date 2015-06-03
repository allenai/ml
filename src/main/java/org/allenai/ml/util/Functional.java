package org.allenai.ml.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple functional operations on collections.
 */
public class Functional {

    private Functional() {
        // intentional no-op
    }

    /**
     * Return lists of lists of equal size (except for the last which will be remainder).
     * Will use `List.sublist` so references to underlying list will remain. If you want
     * a fresh copy, it's caller's responsibility to do so.
     */
    public static <T> List<List<T>> partition(List<T> elems, int numPartitions) {
        if (elems.size() < numPartitions) {
            throw new IllegalArgumentException("Must have more elems than partitions");
        }
        List<List<T>> parts = new ArrayList<>(numPartitions);
        int numPerPart = elems.size() / numPartitions;
        for (int idx = 0; idx < numPartitions; idx++) {
            int start = idx * numPerPart;
            int end = idx+1 < numPartitions ? (idx+1) * numPerPart : elems.size();
            parts.add(elems.subList(start, end));
        }
        return parts;
    }
}
