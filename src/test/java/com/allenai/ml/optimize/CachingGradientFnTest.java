package com.allenai.ml.optimize;

import com.allenai.ml.linalg.DenseVector;
import com.allenai.ml.linalg.Vector;
import lombok.val;
import org.testng.annotations.Test;

import java.util.function.Function;

import static org.testng.AssertJUnit.assertEquals;

@Test
public class CachingGradientFnTest {
    class CountingFn<T,R> implements Function<T,R> {
        private int callCount = 0;
        private final Function<T,R> fn;
        CountingFn(Function<T, R> fn) {
      this.fn = fn;
    }
        @Override
        public R apply(T t) {
            callCount++;
            return fn.apply(t);
        }
    }

    @Test
    public void testCachingGradientFn() throws Exception {
        CountingFn<Vector, GradientFn.Result> countXSquared = new CountingFn<>(TestUtils.xSquared.fn);
        val cacheGradFn = new CachingGradientFn(1, GradientFn.from(1, countXSquared));
        // Underlying fn should only be called once
        cacheGradFn.apply(DenseVector.of(1.0));
        assertEquals(countXSquared.callCount, 1);
        cacheGradFn.apply(DenseVector.of(1.0));
        assertEquals("Result should be cached",countXSquared.callCount, 1);
        // Replace cache should trigger fresh call
        cacheGradFn.apply(DenseVector.of(2.0));
        cacheGradFn.apply(DenseVector.of(1.0));
        assertEquals("Cache should be cleared",countXSquared.callCount, 3);
        cacheGradFn.apply(DenseVector.of(1.0));
        assertEquals("Result should be cached",countXSquared.callCount, 3);
  }
}