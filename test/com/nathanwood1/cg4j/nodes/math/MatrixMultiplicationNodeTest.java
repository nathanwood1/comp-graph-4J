package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.InputNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixMultiplicationNodeTest {
    @Test
    void evaluateNN() {
        InputNode a = new InputNode(new int[]{-1, -1});
        InputNode b = new InputNode(new int[]{-1, -1});
        Node out = new MatrixMultiplicationNode(a, b, false, false);
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 2, 3,
                            4, 5, 6
                    }, new int[]{2, 3}))
                    .addInput(b, new Tensor(new float[]{
                            7, 8,
                            9, 10,
                            11, 12
                    }, new int[]{3, 2}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            58, 64,
                            139, 154
                    }, new int[]{2, 2})
            ));
        }
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 10, 11, 12,
                            13, 14, 15, 16
                    }, new int[]{4, 4}))
                    .addInput(b, new Tensor(new float[]{
                            17, 18, 19, 20,
                            21, 22, 23, 24,
                            25, 26, 27, 28,
                            29, 30, 31, 32
                    }, new int[]{4, 4}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            250, 260, 270, 280,
                            618, 644, 670, 696,
                            986, 1028, 1070, 1112,
                            1354, 1412, 1470, 1528
                    }, new int[]{4, 4})
            ));
        }
    }

    @Test
    void evaluateYN() {
        InputNode a = new InputNode(new int[]{-1, -1});
        InputNode b = new InputNode(new int[]{-1, -1});
        Node out = new MatrixMultiplicationNode(a, b, true, false);
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 4,
                            2, 5,
                            3, 6
                    }, new int[]{3, 2}))
                    .addInput(b, new Tensor(new float[]{
                            7, 8,
                            9, 10,
                            11, 12
                    }, new int[]{3, 2}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            58, 64,
                            139, 154
                    }, new int[]{2, 2})
            ));
        }
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 10, 11, 12,
                            13, 14, 15, 16
                    }, new int[]{4, 4}))
                    .addInput(b, new Tensor(new float[]{
                            17, 18, 19, 20,
                            21, 22, 23, 24,
                            25, 26, 27, 28,
                            29, 30, 31, 32
                    }, new int[]{4, 4}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            724, 752, 780, 808,
                            816, 848, 880, 912,
                            908, 944, 980, 1016,
                            1000, 1040, 1080, 1120
                    }, new int[]{4, 4})
            ));
        }
    }

    @Test
    void evaluateNY() {
        InputNode a = new InputNode(new int[]{-1, -1});
        InputNode b = new InputNode(new int[]{-1, -1});
        Node out = new MatrixMultiplicationNode(a, b, false, true);
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 2, 3,
                            4, 5, 6
                    }, new int[]{2, 3}))
                    .addInput(b, new Tensor(new float[]{
                            7, 9, 11,
                            8, 10, 12
                    }, new int[]{2, 3}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            58, 64,
                            139, 154
                    }, new int[]{2, 2})
            ));
        }
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 10, 11, 12,
                            13, 14, 15, 16
                    }, new int[]{4, 4}))
                    .addInput(b, new Tensor(new float[]{
                            17, 18, 19, 20,
                            21, 22, 23, 24,
                            25, 26, 27, 28,
                            29, 30, 31, 32
                    }, new int[]{4, 4}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            190, 230, 270, 310,
                            486, 590, 694, 798,
                            782, 950, 1118, 1286,
                            1078, 1310, 1542, 1774
                    }, new int[]{4, 4})
            ));
        }
    }

    @Test
    void evaluateYY() {
        InputNode a = new InputNode(new int[]{-1, -1});
        InputNode b = new InputNode(new int[]{-1, -1});
        Node out = new MatrixMultiplicationNode(a, b, true, true);
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 4,
                            2, 5,
                            3, 6
                    }, new int[]{3, 2}))
                    .addInput(b, new Tensor(new float[]{
                            7, 9, 11,
                            8, 10, 12
                    }, new int[]{2, 3}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            58, 64,
                            139, 154
                    }, new int[]{2, 2})
            ));
        }
        {
            Eval e = new Eval()
                    .addInput(a, new Tensor(new float[]{
                            1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 10, 11, 12,
                            13, 14, 15, 16
                    }, new int[]{4, 4}))
                    .addInput(b, new Tensor(new float[]{
                            17, 18, 19, 20,
                            21, 22, 23, 24,
                            25, 26, 27, 28,
                            29, 30, 31, 32
                    }, new int[]{4, 4}));
            assertTrue(e.evaluate(out).equals(
                    new Tensor(new float[]{
                            538, 650, 762, 874,
                            612, 740, 868, 996,
                            686, 830, 974, 1118,
                            760, 920, 1080, 1240
                    }, new int[]{4, 4})
            ));
        }
    }

}