package com.github.cg4j;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Random;

class TensorTest {
    @Test
    void getShape() {
        Tensor tensor = new Tensor(new float[4 * 2 * 5 * 2], new int[]{4, 2, 5, 2});
        Assertions.assertArrayEquals(new int[]{4, 2, 5, 2}, tensor.getShape());
    }

    @Test
    void getSetValI() {
        int[] shape = {4, 3, 2};
        float[] vals = new float[4 * 3 * 2];

        vals[2] = 1;
        vals[16] = 1;
        vals[23] = 1;

        Tensor tensor = new Tensor(vals, shape);

        for (int i = 0; i < 4 * 3 * 2; i++) {
            Assertions.assertEquals(vals[i], tensor.getVal(i));
            float random = (float) Math.random();
            tensor.setVal(i, random);
            Assertions.assertEquals(random, tensor.getVal(i));
        }
    }

    @Test
    void getSetValIs() {
        int[] shape = {4, 3, 2};
        float[] vals = new float[4 * 3 * 2];

        vals[2] = 1;
        vals[16] = 1;
        vals[23] = 1;

        Tensor tensor = new Tensor(vals, shape);

        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 2; z++) {
                    Assertions.assertEquals(vals[x * 3 * 2 + y * 2 + z], tensor.getVal(new int[]{x, y, z}));
                    float random = (float) Math.random();
                    tensor.setVal(new int[]{x, y, z}, random);
                    Assertions.assertEquals(random, tensor.getVal(new int[]{x, y, z}));
                }
            }
        }
    }

    @Test
    void indicesToAndFromIndex() {
        Tensor tensor = Tensor.fromRandom(-1, 1, new int[]{4, 3, 2});

        for (int i = 0; i < 4 * 3 *  2; i++) {
            Assertions.assertEquals(i, tensor.getIndexFromIndices(tensor.getIndicesFromIndex(i)));
            Assertions.assertEquals(tensor.getVal(i), tensor.getVal(tensor.getIndicesFromIndex(i)));
        }
    }

    @Test
    void fromRandomObject() {
        Random r = new Random(0);
        Tensor t = Tensor.fromRandom(r, -5, 3, new int[]{4, 6, 2});

        r.setSeed(0);
        Tensor t2 = Tensor.fromRandom(r, -5, 3, new int[]{4, 6, 2});
        Tensor t3 = Tensor.fromRandom(r, -5, 3, new int[]{4, 6, 2});
        Assertions.assertTrue(t.equals(t2));
        Assertions.assertFalse(t.equals(t3));

        Assertions.assertEquals(4 * 6 * 2, t.length);
        Assertions.assertArrayEquals(new int[]{4, 6, 2}, t.getShape());

        float lowest = 100;
        float highest = -100;
        for (int i = 0; i < t.length; i++) {
            lowest = Math.min(lowest, t.getVal(i));
            highest = Math.max(highest, t.getVal(i));
        }
        Assertions.assertTrue(Math.round(lowest) == -5);
        Assertions.assertTrue(Math.round(highest) == 3);
    }

    @Test
    void fromRandomMath() {
        Tensor t = Tensor.fromRandom(10, 49, new int[]{6, 6, 2, 10, 10});

        Assertions.assertEquals(6 * 6 * 2 * 10 * 10, t.length);
        Assertions.assertArrayEquals(new int[]{6, 6, 2, 10, 10}, t.getShape());

        float lowest = 100;
        float highest = -100;
        for (int i = 0; i < t.length; i++) {
            lowest = Math.min(lowest, t.getVal(i));
            highest = Math.max(highest, t.getVal(i));
        }
        Assertions.assertTrue(Math.round(lowest) == 10);
        Assertions.assertTrue(Math.round(highest) == 49);
    }

    @Test
    void toStringTest() {
        Tensor tensor = new Tensor(new float[]{
                0, 1,
                2, 3,
                4, 5,

                6, 7,
                8, 9,
                10, 11
        }, new int[]{2, 3, 2});

        Assertions.assertTrue("Tensor(Shape=[2, 3, 2], Data=[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]])".equals(tensor.toString()));
    }

    @Test
    void arrayToStringTest() {
        Tensor tensor = new Tensor(new float[]{
                0, 1,
                2, 3,
                4, 5,

                6, 7,
                8, 9,
                10, 11
        }, new int[]{2, 3, 2});

        Assertions.assertTrue("[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]]".equals(tensor.arrayToString()));
    }

}