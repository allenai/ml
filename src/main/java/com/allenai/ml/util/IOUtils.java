package com.allenai.ml.util;

import lombok.SneakyThrows;
import lombok.val;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
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
            dos.writeDouble(xs[idx]);
        }
    }

    @SneakyThrows
    public static double[] loadDoubles(DataInputStream dis) {
        int n = dis.readInt();
        double[] xs = new double[n];
        for (int idx = 0; idx < n; idx++) {
            xs[idx] = dis.readDouble();
        }
        return xs;
    }

    @SneakyThrows
    public static Stream<String> linesFromPath(String path) {
        return new BufferedReader(new FileReader(path)).lines();
    }

    @SneakyThrows
    public static void saveList(DataOutputStream dos, List<String> elems) {
        dos.writeInt(elems.size());
        for (String elem : elems) {
            dos.writeUTF(elem);
        }
    }

    @SneakyThrows
    public static List<String> loadList(DataInputStream dis) {
        int n = dis.readInt();
        val lst = new ArrayList<String>(n);
        for (int idx = 0; idx < n; idx++) {
            lst.add(dis.readUTF());
        }
        return lst;
    }
}
