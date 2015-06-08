package org.allenai.ml.util;

import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnel;
import com.gs.collections.api.map.primitive.ObjectIntMap;
import com.gs.collections.impl.map.mutable.primitive.ObjectIntHashMap;
import javafx.scene.effect.Bloom;
import lombok.val;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A list that implements O(1) random access lookup AND O(1) <code>indexOf</code> calls. It also
 * does not permit duplicate elements.
 * @param <T>
 */
public class Indexer<T> extends AbstractList<T> {

    private final List<T> list;
    private final ObjectIntMap<T> objToIndex;
    private BloomFilter filter;

    private Indexer(Stream<T> elems) {
        this.list = elems
            .distinct()
            .collect(Collectors.toList());
        val m = new ObjectIntHashMap<T>();
        for (int idx = 0; idx < list.size(); idx++) {
            m.put(list.get(idx), idx);
        }
        this.objToIndex = m.toImmutable();
    }

    /**
     * Put a bloom filter in front of the `indexOf` call. Useful for cases
     * when you expect a lot of out-of-indexer calls.
     * @param expectedFailureProb
     */
    public void addBloomFilter(double expectedFailureProb) {
        Funnel<T> funnel = (t, sink) -> {
            sink.putInt(t.hashCode());
        };
        this.filter = BloomFilter.create(funnel, size(), expectedFailureProb);
        for (T t : list) {
            this.filter.put(t);
        }
    }

    @Override
    public int size() {
        return list.size();
    }

    @Override
    public boolean isEmpty() {
        return list.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
        return objToIndex.containsKey(o);
    }

    @Override
    public Iterator<T> iterator() {
        return list.iterator();
    }

    @Override
    public Object[] toArray() {
        return list.toArray();
    }

    @Override
    public boolean add(T t) {
        throw new RuntimeException("Indexer is immutable and doesn't support add()");
    }

    @Override
    public T get(int index) {
        return list.get(index);
    }

    @Override
    public int indexOf(Object o) {
        if (filter != null && !filter.mightContain(o)) {
            return -1;
        }
        return objToIndex.get(o);
    }

    @Override
    public int lastIndexOf(Object o) {
        return indexOf(o);
    }


    @Override
    public List<T> subList(int fromIndex, int toIndex) {
        return list.subList(fromIndex, toIndex);
    }


    public static <T> Indexer<T> fromStream(Stream<T> stream) {
        return new Indexer<>(stream);
    }

    private final static String DATA_VERSION = "1.0";

    public void save(DataOutputStream dos) throws IOException {
        dos.writeUTF(DATA_VERSION);
        IOUtils.saveList(dos, this.stream().map(Object::toString).collect(Collectors.toList()));
    }

    public static Indexer<String> load(DataInputStream dis) throws IOException {
        IOUtils.ensureVersionMatch(dis, DATA_VERSION);
        val lst = IOUtils.loadList(dis);
        return new Indexer<>(lst.stream());
    }
}
