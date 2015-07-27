package org.allenai.ml.sequences.crf;

import org.allenai.ml.linalg.Vector;
import com.gs.collections.api.list.primitive.*;
import com.gs.collections.impl.list.mutable.primitive.DoubleArrayList;
import com.gs.collections.impl.list.mutable.primitive.IntArrayList;
import lombok.RequiredArgsConstructor;
import lombok.val;

import java.util.List;

/**
 * Memory-efficient representation of the predicates (feature tempaltes) on the node
 * and edge cliques in the chain CRF model. This class already assumes the data has already
 * been indexed so the predicates are represented by a `Vector`.
 *
 * This class can either represent a training or inference example, depending on whether
 * `goldLabels` states have been passed in.
 *
 * __Internal notes__: Rather than storing a separate `Vector` for each node/edge in the sequence,
 * we store a long flat integer and double array of all the vectors along with offsets to know the boundaries
 * for various positions. Since there is about 128 bytes overhead per Vector (class overhead plus some other member
 * variables), this can be a substantial fraction of the overall memory footprint if you only have a few dozen features.
 */
public class CRFIndexedExample {

    // First all the node predicates and then edge predicates
    private final ImmutableIntList allPredicateIndices;
    private final ImmutableDoubleList allPredicateValues;
    private final ImmutableIntList offsets;
    private final int sequenceLength;
    private final ImmutableIntList goldLabels;

    public CRFIndexedExample(List<Vector> nodePredicates, List<Vector> edgePredicates, int[] goldLabels) {
        assert nodePredicates.size() == edgePredicates.size()+1;
        assert goldLabels == null || goldLabels.length == nodePredicates.size();
        // Append to indices/values for node/edge predicates separately
        val allPredicateIndices = new IntArrayList();
        val allPredicateValues = new DoubleArrayList();
        val nodeOffsets = flattenPredicates(nodePredicates, allPredicateIndices, allPredicateValues, 0);
        int numNodePredicates = allPredicateIndices.size();
        val edgeOffsets = flattenPredicates(edgePredicates, allPredicateIndices, allPredicateValues, numNodePredicates);
        val offsets = new IntArrayList(nodeOffsets.size() + edgeOffsets.size());
        offsets.addAll(nodeOffsets);
        offsets.addAll(edgeOffsets);

        this.sequenceLength = nodePredicates.size();
        this.offsets = offsets.toImmutable();
        this.allPredicateIndices = allPredicateIndices.toImmutable();
        this.allPredicateValues = allPredicateValues.toImmutable();
        this.goldLabels = goldLabels != null ? new IntArrayList(goldLabels).toImmutable() : null;
    }

    public CRFIndexedExample(List<Vector> nodePredicates, List<Vector> edgePredicates) {
        this(nodePredicates, edgePredicates, null);
    }

    private IntList flattenPredicates(List<Vector> predicateVectors,
                                      MutableIntList predIndices,
                                      MutableDoubleList predVals,
                                      int startOffset) {
        val offsets = new IntArrayList(predicateVectors.size());
        int totalOffset = startOffset;
        // bottleneck: Low-level intentional here
        for (int idx = 0; idx < predicateVectors.size(); idx++) {
            offsets.add(totalOffset);
            val it = predicateVectors.get(idx).iterator();
            while (!it.isExhausted()) {
                // Don't bother writing 0.0 valued features
                if (it.value() != 0.0) {
                    predIndices.add((int)it.index());
                    predVals.add(it.value());
                    totalOffset ++;
                }
                it.advance();
            }
        }
        return offsets;
    }

    public int[] getGoldLabels() {
        return goldLabels.toArray();
    }

    public boolean isLabeled() {
        return goldLabels != null;
    }

    @RequiredArgsConstructor
    private class Iterator implements Vector.Iterator {
        private final int start, stop;
        private int offset = 0;

        @Override
        public boolean isExhausted() {
            return offset >= stop - start;
        }

        @Override
        public void advance() {
            offset++;
        }

        private void ensureNotExhausted() {
            if (isExhausted()) {
                throw new RuntimeException("Iterator is exhausted");
            }
        }

        @Override
        public long index() {
            ensureNotExhausted();
            return allPredicateIndices.get(start + offset);
        }

        @Override
        public double value() {
            ensureNotExhausted();
            return allPredicateValues.get(start + offset);
        }

        @Override
        public void reset() {
            offset = 0;
        }
    }

    public Vector.Iterator getNodePredicateValues(int idx) {
        if (idx >= getSequenceLength()) {
            throw new IllegalArgumentException("Invalid node predicate index");
        }
        int start = offsets.get(idx);
        int stop = offsets.get(idx + 1);
        return new Iterator(start, stop);
    }

    public Vector.Iterator getEdgePredicateValues(int idx) {
        if (idx >= getSequenceLength()-1) {
            throw new IllegalArgumentException("Invalid node transition edge index");
        }
        // edge offsets after node ones
        int start = offsets.get(getSequenceLength() + idx);
        int stopIdx = getSequenceLength() + idx + 1;
        int stop =  stopIdx < offsets.size() ? offsets.get(stopIdx) : allPredicateIndices.size();
        return new Iterator(start, stop);
    }

    public int getSequenceLength() {
        return sequenceLength;
    }
}
