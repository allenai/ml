package org.allenai.ml.linalg;

import lombok.val;
import org.testng.annotations.Test;

import java.util.List;

import static java.util.stream.Collectors.toList;
import static org.testng.Assert.assertEquals;

@Test
public class DenseVectorTest {

    public void testDimension() throws Exception {
        DenseVector v = DenseVector.of(10);
        assertEquals(v.dimension(), 10);
    }

    public void testAt() throws Exception {
        double[] elems = {1.0,2.0,3.0};
        DenseVector v = DenseVector.of(elems);
        assertEquals(v.at(1), 2.0);
    }

    public void testSet() throws Exception {
        DenseVector v = DenseVector.of(3);
        v.set(2, 1.0);
        assertEquals(v.at(2), 1.0);
    }

    public void testNonZeroEntries() throws Exception {
        double[] elems = {1.0,2.0,3.0};
        val v = DenseVector.of(elems);
        List<Vector.Entry> entries =  v.nonZeroEntries().collect(toList());
        assertEquals(entries.get(0), Vector.Entry.of(0, 1.0));
    }

    public void testNumStoredEntries() throws Exception {
        DenseVector v = DenseVector.of(5);
        assertEquals(v.numStoredEntries(), 5);
    }

    public void testCopy() throws Exception {
        val v = DenseVector.of(new double[]{1.0,2.0,3.0});
        VectorTest.testCopy(v);
    }

    public void testDotProduct() throws Exception {
        val vec = DenseVector.of(new double[]{1.0,2.0,3.0});
        assertEquals(vec.dotProduct(vec), 14.0);
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testLookupException() {
        val vec = DenseVector.of(new double[]{1.0,2.0,3.0});
        vec.at(5);
    }
}