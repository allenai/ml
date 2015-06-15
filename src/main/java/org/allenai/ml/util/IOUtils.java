package org.allenai.ml.util;

import lombok.SneakyThrows;
import lombok.val;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileReader;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class IOUtils {

    @SneakyThrows
    public static void ensureVersionMatch(DataInputStream dis, String expectedVersion) {
        String actualVersion = dis.readUTF();
        if (!actualVersion.equals(expectedVersion)) {
            throw new IllegalArgumentException(String.format(
                "Data versions't don't match. Saved is %s but current code is %s",
                actualVersion, expectedVersion));
        }
    }

    @SneakyThrows
    public static void saveDoubles(DataOutputStream dos, double[] xs) {
        dos.writeInt(xs.length);
        for (int idx = 0; idx < xs.length; idx++) {
            dos.writeFloat((float) xs[idx]);
        }
    }

    @SneakyThrows
    public static double[] loadDoubles(DataInputStream dis) {
        int n = dis.readInt();
        double[] xs = new double[n];
        for (int idx = 0; idx < n; idx++) {
            xs[idx] = dis.readFloat();
        }
        return xs;
    }

    @SneakyThrows
    public static Stream<String> linesFromPath(String path) {
        return new BufferedReader(new FileReader(path)).lines();
    }

    private final static int INDEXER_BLOCK_SIZE = 1000;

    @SneakyThrows
    public static void saveList(DataOutputStream dos, List<String> elems) {
        dos.writeInt(elems.size());
        Charset utf8 = Charset.forName("UTF8");
        for (String elem: elems) {
            byte[] elemBytes = elem.getBytes();
            dos.writeInt(elemBytes.length);
            dos.write(elemBytes);
        }
    }

    @SneakyThrows
    public static List<String> loadList(DataInputStream dis) {
        int n = dis.readInt();
        Charset utf8 = Charset.forName("UTF8");
        val lst = new ArrayList<String>(n);
        for (int idx = 0; idx < n; idx++) {
            int strLen = dis.readInt();
            byte[] elemBytes = new byte[strLen];
            dis.read(elemBytes);
            String elem = new String(elemBytes, utf8);
            lst.add(elem);
        }
        return lst;
    }
}
