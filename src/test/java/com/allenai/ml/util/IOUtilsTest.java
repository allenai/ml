package com.allenai.ml.util;

import lombok.val;
import org.testng.annotations.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Function;

import static org.testng.Assert.*;

@Test
public class IOUtilsTest {

    private static <T> void testRoundtrip(
            BiConsumer<DataOutputStream, T> writeFn,
            Function<DataInputStream, T> readFn,
            T elem) {
        val baos = new ByteArrayOutputStream(3200);
        val dos = new DataOutputStream(baos);
        writeFn.accept(dos, elem);
        byte[] bs = baos.toByteArray();
        val dis = new DataInputStream(new ByteArrayInputStream(bs));
        T output = readFn.apply(dis);
        assertEquals(elem, output);
    }

    public void testRoundtripDoubles() {
        double[] xs = new double[]{1.0, 2.0, 3.0};
        testRoundtrip(IOUtils::saveDoubles, IOUtils::loadDoubles, xs);
    }

    public void testRoundtripElems() {
        List<String> elems = Arrays.asList("a", "b", "c");
        testRoundtrip(IOUtils::saveList, IOUtils::loadList, elems);
    }
}