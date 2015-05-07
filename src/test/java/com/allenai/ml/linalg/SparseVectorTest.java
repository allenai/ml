package com.allenai.ml.linalg;

import org.testng.annotations.Test;

import static org.testng.Assert.assertEquals;

@Test
public class SparseVectorTest {

    public void testDotProduct() throws Exception {
        Vector sparse = SparseVector.make(3);
        Vector dense = DenseVector.of(new double[]{1.0,2.0,3.0});
        sparse.set(0, 1.0);
        sparse.set(1, 2.0);
        assertEquals(sparse.dotProduct(dense), 5.0);
    }

    public void testCopy() throws Exception {
        Vector sparse = SparseVector.make(3);
        sparse.set(0, 1.0);
        VectorTest.testCopy(sparse);
    }
}