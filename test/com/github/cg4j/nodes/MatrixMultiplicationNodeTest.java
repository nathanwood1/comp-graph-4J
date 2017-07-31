package com.github.cg4j.nodes;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class MatrixMultiplicationNodeTest {
    @Test
    void getShape() {
        InputNode a = new InputNode(new int[]{-1, 6});
        InputNode b = new InputNode(new int[]{6, 51});
        MatrixMultiplicationNode c = new MatrixMultiplicationNode(a, b);
        Assertions.assertArrayEquals(new int[]{-1, 51}, c.getShape());
    }

    @Test
    void evaluate1() {
        InputNode a = new InputNode(new int[]{-1, 3});
        InputNode b = new InputNode(new int[]{3, 2});
        MatrixMultiplicationNode c = new MatrixMultiplicationNode(a, b);

        Tensor aInput = new Tensor(new float[]{
                1, 2,
                3, 4,
                5, 6
        }, new int[]{2, 3});
        Tensor bInput = new Tensor(new float[]{
                7, 8, 9,
                10, 11, 12
        }, new int[]{3, 2});

        Eval eval = new Eval()
                .addInput(a, aInput)
                .addInput(b, bInput);
        Assertions.assertTrue(new Tensor(new float[]{
                58, 64,
                139, 154
        }, new int[]{2, 2}).equals(eval.evaluate(c)));
    }

    @Test
    void evaluate2() {
        InputNode a = new InputNode(new int[]{-1, 2, 3});
        InputNode b = new InputNode(new int[]{2, 3, 2});
        MatrixMultiplicationNode c = new MatrixMultiplicationNode(a, b);

        Tensor aInput = new Tensor(new float[]{
                1, 2, 3,
                4, 5, 6,

                7, 8, 9,
                10, 11, 12
        }, new int[]{2, 2, 3});
        Tensor bInput = new Tensor(new float[]{
                7, 8,
                9, 10,
                11, 12,

                13, 14,
                15, 16,
                17, 18
        }, new int[]{2, 3, 2});

        Eval eval = new Eval()
                .addInput(a, aInput)
                .addInput(b, bInput);
        Assertions.assertTrue(new Tensor(new float[]{
                58, 64,
                94, 100,

                139, 154,
                229, 244,

                220, 244,
                364, 388,

                301, 334,
                499, 532
        }, new int[]{2, 2, 2, 2}).equals(eval.evaluate(c)));
    }

}