package org.allenai.ml.linalg;

import org.allenai.ml.util.Indexer;
import com.gs.collections.api.map.primitive.LongDoubleMap;
import com.gs.collections.api.map.primitive.MutableLongDoubleMap;
import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.tuple.primitive.LongDoublePair;
import com.gs.collections.impl.map.mutable.primitive.LongDoubleHashMap;
import lombok.val;

import java.util.stream.LongStream;
import java.util.stream.Stream;

public class SparseVector implements Vector {

    private final MutableLongDoubleMap vec;
    private final long dimension;

    private SparseVector(MutableLongDoubleMap vec, long dimension) {
      this.vec = vec;
      this.dimension = dimension;
    }

    public static SparseVector withCapacity(int capacity, long dimension) {
      return new SparseVector(new LongDoubleHashMap(capacity), dimension);
    }

    public static SparseVector fromEntries(Stream<Entry> entries, long dimension) {
      val vec = new LongDoubleHashMap();
      entries.forEach(e -> vec.put(e.index, e.value));
      return new SparseVector(vec, dimension);
    }

    public static SparseVector make(long dimension) {
      return new SparseVector(new LongDoubleHashMap(), dimension);
    }

    public static SparseVector make(LongDoubleMap vec, long dimension) {
      return new SparseVector(new LongDoubleHashMap(vec), dimension);
    }

    public static SparseVector make() {
      return make(Long.MAX_VALUE);
    }

    @Override
    public long dimension() {
      return dimension;
    }

    @Override
    public double at(long dimensionIdx) {
      return vec.get(dimensionIdx);
    }

    @Override
    public void set(long dimensionIdx, double val) {
      vec.put(dimensionIdx, val);
    }

    @Override
    public long numStoredEntries() {
      return vec.size();
    }

    @Override
    public Vector copy() {
      return new SparseVector(new LongDoubleHashMap(vec), dimension);
    }

    @Override
    public Stream<Entry> nonZeroEntries() {
      long[] indices = vec.keySet().toArray();
      return LongStream.of(indices)
          .mapToObj(idx -> Vector.Entry.of(idx, vec.get(idx)));
    }

    @Override
    public Vector.Iterator iterator() {
        long[] indices = vec.keysView().toArray();
        return new Vector.Iterator() {
            int offset = 0;

            public boolean isExhausted() {
                return offset >= indices.length;
            }

            @Override
            public void reset() {
                offset = 0;
            }

            @Override
            public void advance() {
                offset ++;
            }

            @Override
            public long index() {
                return indices[offset];
            }

            @Override
            public double value() {
                return at(indices[offset]);
            }
        };
    }

    @Override
    public double dotProduct(Vector other) {
      if (other.numStoredEntries() < this.numStoredEntries()) {
          return other.dotProduct(this);
      }
      double result = 0.0;
      val it = vec.keyValuesView().iterator();
      while (it.hasNext()) {
          LongDoublePair pair = it.next();
          long idx = pair.getOne();
          double val = pair.getTwo();
          result += val * other.at(idx);
      }
      return result;
    }

    public static <T extends Comparable<T>> SparseVector indexed(ObjectDoubleMap<T> map, Indexer<T> indexer) {
        val ldm = new LongDoubleHashMap(map.size());
        map.forEachKeyValue((k, v) -> {
            int idx = indexer.indexOf(k);
            if (idx >= 0) {
                ldm.put(idx, v);
            }
        });
        // since we made this map, no concern over someone
        // else mutating, can use direct ctor
        return new SparseVector(ldm, indexer.size());
    }
}
