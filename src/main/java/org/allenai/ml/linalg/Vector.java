package org.allenai.ml.linalg;

import lombok.Data;
import lombok.val;

import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Interface for a linear algebra vector. Can represent sparse and dense vectors. Reasonable amount of care taken
 */
public interface Vector {
    /**
     *
     *  @return dimension of the space for the vector
     */
    long dimension();

    /**
     *
     * @param dimensionIdx
     * @return value at `dimensionIdx`
     */
    double at(long dimensionIdx);

    /**
     * Set value at `dimensionIdx`
     * @param dimensionIdx
     * @param val
     */
    void set(long dimensionIdx, double val);

    /**
     * @return How many entries are stored explicitly in the Vector
     */
    long numStoredEntries();

    /**
     * @return a copy of the vector you can freely mutate
     */
    Vector copy();

    /**
     * For use cases when you don't want to allocate `Vector.Entry` on a per-entry basis and want to re-use the iterator
     * without object allocation. This interface allows for an implementation where there is no object allocation
     * associated with iterating over elements or re-using.
     * Example usage:
     * ```java
     * while (it.isExhausted()) {
     *     long idx = it.index();
     *     double val = it.value();
     *     // do stuff with (idx, val)
     *     it.advance();
     * }
     * it.reset(); // can now safely re-use
     * ```
     */
    interface Iterator {
        boolean isExhausted();
        void reset();
        void advance();
        long index();
        double value();
    }

    /**
     * External iterator, useful for when you don't want object allocation
     * to access individual elements (an internal iterator would require
     * creating a pair object for the index/value).
     */
    Iterator iterator();

    /**
     * @param dimensionIndices
     * @return A slice of the Vector at the specified `dimensionIndices`. The lenght of the slice
     * is the same as `dimensionIndices`.
     */
    default double[] at(long... dimensionIndices) {
        val result = new double[dimensionIndices.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = at(dimensionIndices[i]);
        }
        return result;
    }

    /**
     * Mutate the Vector-in-place according to `updateFn`
     * @param updateFn
     */
    default void mapInPlace(EntryUpdateFunction updateFn) {
        entries().forEachOrdered(entry -> {
            double updatedValue = updateFn.update(entry.index, entry.value);
            set(entry.index, updatedValue);
        });
    }

    /**
     *
     * @param updateFn
     * @return Copy of the Vector with each entry having `updateFn` applied
     */
    default Vector map(EntryUpdateFunction updateFn) {
        Vector result = copy();
        result.mapInPlace(updateFn);
        return result;
    }

    /**
     * In place update of `scale  * x_i + offset` for each element.
     * @param scale
     * @param offset
     */
    default void affineUpdateInPlace(double scale, double offset) {
        this.mapInPlace((idx, value) -> scale * value + offset);
    }

    /**
     * Like [`affineUpdateInPlace`] but
     * @param scale
     * @param offset
     * @return
     */
    default Vector affine(double scale, double offset) {
      Vector result = copy();
      result.affineUpdateInPlace(scale, offset);
      return result;
    }

    /**
     * @param scale
     * @return Copy of the vector scaled by `scale`
     */
    default Vector scale(double scale) {
        return affine(scale, 0.0);
    }

    /**
    * Compute the [dot-product](http://en.wikipedia.org/wiki/Dot_product) with `other` vector
    * @param other
    * @return
    */
    default double dotProduct(Vector other) {
      if (other.dimension() != this.dimension()) {
          throw new IllegalArgumentException("Dimensions don't match");
      }
      // Do dot-product from the pov of vector with fewer elements
      if (other.numStoredEntries() < this.numStoredEntries()) {
        return other.dotProduct(this);
      }
      return nonZeroEntries()
        .mapToDouble(entry -> entry.value * other.at(entry.index))
        .sum();
    }

    default double l2NormSquared() {
        return this.dotProduct(this);
    }

    default double l2Distance(Vector other) {
        return this.add(-1.0, other).l2NormSquared();
    }

    default boolean closeTo(Vector other, double thresh) {
        return this.l2Distance(other) < thresh;
    }

    double DIST_THRESH = 1.0e-10;

    default boolean closeTo(Vector other) {
        return closeTo(other, DIST_THRESH);
    }


    default double inc(long idx, double amount) {
        double newValue = at(idx) + amount;
        set(idx, newValue);
        return newValue;
    }

    default Stream<Entry> entries() {
        val spliterator = new VectorSpliterator(this, 0, dimension());
        return StreamSupport.stream(spliterator, true);
    }

    default Stream<Entry> nonZeroEntries() {
        return entries().filter(e -> e.value != 0.0);
    }

    default Vector add(double scale, Vector dir) {
        Vector result = this.copy();
        result.addInPlace(scale, dir);
        return result;
    }

    default void addInPlace(double scale, Vector dir) {
        Vector.Iterator it = dir.iterator();
        while (!it.isExhausted()) {
            inc(it.index(), scale * it.value());
            it.advance();
        }
    }

    default Vector add(Vector dir) {
        return add(1.0, dir);
    }

    default void scaleInPlace(double v) {
        this.nonZeroEntries().forEach(e -> this.set(e.getIndex(), v * e.getValue()));
    }

    default double[] toDoubles() {
        int dim = (int) dimension();
        double[] xs = new double[dim];
        for (int idx = 0; idx < dim; idx++) {
            xs[idx] = at(idx);
        }
        return xs;
    }

    /**
     * Add an iterator's entries to the receiver, returning self
     * @param scale
     * @param iter
     * @return `this` for chaining
     */
    default Vector addInPlace(double scale, Vector.Iterator iter) {
        while (!iter.isExhausted()) {
            long idx = iter.index();
            this.set(idx, this.at(idx) + scale * iter.value());
            iter.advance();
        }
        return this;
    }

    default Vector addInPlace(Vector.Iterator iter) {
        return addInPlace(1.0, iter);
    }


    @Data(staticConstructor = "of")
    final class Entry implements Comparable<Entry> {
        public final long index;
        public final double value;

        @Override
        public int compareTo(Entry o) {
            return Long.compare(this.index, o.index);
        }
    }

    final class VectorSpliterator implements Spliterator<Entry> {
        private Vector vec;
        private long position;
        private long stop;

        VectorSpliterator(Vector vec, long start, long stop) {
            this.vec = vec;
            this.position = start;
            this.stop = stop;
        }

        @Override
        public boolean tryAdvance(Consumer<? super Entry> action) {
            if (position < stop) {
               action.accept(Entry.of(position, vec.at(position)));
                position++;
                return true;
            }
            return false;
        }

        @Override
        public Spliterator<Entry> trySplit() {
            long mid = (stop-position)/2 + position;
            if (position < mid) {
                long low = position;
                this.position = mid;
                return new VectorSpliterator(vec,low, mid);
            }
            return null;
        }

        @Override
        public long estimateSize() {
            return stop - position;
        }

        @Override
        public int characteristics() {
            return ORDERED | SIZED | IMMUTABLE | SUBSIZED;
        }
    }

    @FunctionalInterface
    interface EntryUpdateFunction {
        double update(long dimensionIdx, double value);
    }
}
