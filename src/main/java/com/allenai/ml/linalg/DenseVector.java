package com.allenai.ml.linalg;

import lombok.val;

import java.util.Arrays;

public class DenseVector implements Vector {
    private final double[] elems;

    private DenseVector(double[] elems) {
        this.elems = elems;
    }

    public static DenseVector of(long numDimensions) {
        return new DenseVector(new double[(int)numDimensions]);
    }

    /**
     * Wrap a `double[]` in a `DenseVector` view. Does not copy the array, mutating the input
     * array will change the resulting vector, use `DenseVector::copyFrom` to get a fresh copy base
     * on the input
     */
    public static DenseVector of(double... elems) {
        return new DenseVector(elems);
    }


    @Override
    public long dimension() {
        return elems.length;
    }

    private static void ensureIndexIsInteger(long dimensionIdx) {
        if (dimensionIdx > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Index overflows integer max");
        }
    }

    @Override
    public double at(long dimensionIdx) {
        if (dimensionIdx >= dimension()) {
            val errMsg = String.format("Illegal index %d > dimension %d",dimensionIdx,dimension());
            throw new IllegalArgumentException(errMsg);
        }
        ensureIndexIsInteger(dimensionIdx);
        return elems[(int) dimensionIdx];
    }

    @Override
    public void set(long dimensionIdx, double val) {
        ensureIndexIsInteger(dimensionIdx);
        elems[(int) dimensionIdx] = val;
    }

    @Override
    public long numStoredEntries() {
        return elems.length;
    }

    @Override
    public Vector copy() {
        return new DenseVector(Arrays.copyOf(elems, elems.length));
    }

    public Vector.Iterator iterator() {
        return new Iterator() {
            int offset = 0;
            @Override
            public boolean isExhausted() {
                return offset >= elems.length;
            }

            @Override
            public void reset() {
                offset = 0;
            }

            @Override
            public void advance() {
                offset++;
            }

            @Override
            public long index() {
                return offset;
            }

            @Override
            public double value() {
                return elems[offset];
            }
        };
    }
}
