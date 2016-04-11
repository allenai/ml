package org.allenai.ml.util;

import lombok.val;
import org.testng.annotations.Test;

import java.io.*;
import java.util.stream.Stream;

import static org.testng.Assert.*;

@Test
public class IndexerTest {

    private final static Indexer<String> avengers =
        Indexer.fromStream(Stream.of("cap", "iron-man", "hulk", "cap"));

    public void testIndexer() {
        assertFalse( avengers.isEmpty() );
        assertTrue( avengers.size() == 3 );
        assertTrue( avengers.indexOf("cap") == 0 );
        assertTrue( avengers.lastIndexOf("cap") == 0 );
        assertTrue( avengers.get(0).equals("cap") );
        assertTrue( avengers.contains("cap") );
        assertFalse(avengers.contains("made-up"));
        Object[] arr = avengers.toArray();
        assertEquals( arr, new String[]{"cap", "hulk", "iron-man"} );
        assertEquals( avengers, avengers.subList(0, avengers.size()) );

    }

    @Test(expectedExceptions = RuntimeException.class)
    public void testIndexerThrowsRemove() {
        Indexer.fromStream(Stream.of("a", "b", "c")).remove(0);
    }

    @Test(expectedExceptions = RuntimeException.class)
    public void testIndexerThrowsAdd() {
        Indexer.fromStream(Stream.of("a", "b", "c")).add("d");
    }

    public void testSaveLoadRoundtrip() throws IOException {
        val baos = new ByteArrayOutputStream(3200);
        val dos = new DataOutputStream(baos);
        avengers.save(dos);
        val dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
        val otherAvengers = Indexer.load(dis);
        assertEquals(avengers, otherAvengers);
    }
}