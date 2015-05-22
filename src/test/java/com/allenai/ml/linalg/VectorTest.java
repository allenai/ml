package com.allenai.ml.linalg;

import lombok.val;
import org.junit.Assert;

import java.util.List;

import static java.util.stream.Collectors.toList;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.assertTrue;

public class VectorTest {
    public static void testCopy(Vector v) {
        List<Vector.Entry> vectorEntries = v.nonZeroEntries().sorted().collect(toList());
        val copy = v.copy();
        List<Vector.Entry> copyEntries = copy.nonZeroEntries().sorted().collect(toList());

        assertTrue(v.closeTo(copy));
        assertTrue(v.closeTo(SparseVector.fromEntries(vectorEntries.stream(), v.dimension())));
        // Explicitly check hash code
        assertEquals(vectorEntries.stream().mapToInt(Object::hashCode).toArray(),
            copyEntries.stream().mapToInt(Object::hashCode).toArray());
        assertEquals(vectorEntries, copyEntries);
        assertEquals(v.dotProduct(copy), v.l2NormSquared(), 1.0e-10);
        assertTrue(v.scale(2.0).l2Distance(v.add(v)) < 1.0e-10);
        assertTrue(v.map((idx, x) -> 2.0 * x).closeTo(v.scale(2.0)));

        val firstEntry = copyEntries.get(0);
        copy.inc(firstEntry.getIndex(), 10.0);
        assertEquals(copy.at(firstEntry.getIndex()), firstEntry.getValue() + 10.0);

        long[] indices = vectorEntries.stream().mapToLong(Vector.Entry::getIndex).toArray();
        double[] vals = v.at(indices);
        double[] directVals = vectorEntries.stream().mapToDouble(Vector.Entry::getValue).toArray();
        assertEquals(vals, directVals);
    }
}
