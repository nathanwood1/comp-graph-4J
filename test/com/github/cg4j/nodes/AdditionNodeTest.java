package com.github.cg4j.nodes;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class AdditionNodeTest {
    @Test
    void evaluate1() {
        InputNode a = new InputNode(new int[]{2, 2, 2});
        InputNode b = new InputNode(new int[]{2});
        Node output = new AdditionNode(a, b);

        Tensor aI = new Tensor(new float[]{
                1, 2,
                5, 10,

                5, 2,
                0, 0
        }, new int[]{2, 2, 2});
        Tensor bI = new Tensor(new float[]{
                5, 1,
        }, new int[]{2});
        Eval e = new Eval()
                .addInput(a, aI)
                .addInput(b, bI);
        Assertions.assertTrue(e.evaluate(output).equals(new Tensor(new float[]{
                6, 3,
                10, 11,

                10, 3,
                5, 1
        }, new int[]{2, 2, 2})));
    }

    @Test
    void evaluate2() {
        InputNode a = new InputNode(new int[]{2});
        InputNode b = new InputNode(new int[]{2, 2, 2});
        Node output = new AdditionNode(a, b);

        Tensor aI = new Tensor(new float[]{
                5, 1,
        }, new int[]{2});
        Tensor bI = new Tensor(new float[]{
                1, 2,
                5, 10,

                5, 2,
                0, 0
        }, new int[]{2, 2, 2});
        Eval e = new Eval()
                .addInput(a, aI)
                .addInput(b, bI);
        Assertions.assertTrue(e.evaluate(output).equals(new Tensor(new float[]{
                6, 3,
                10, 11,

                10, 3,
                5, 1
        }, new int[]{2, 2, 2})));
    }

    @Test
    void evaluate3() {
        InputNode a = new InputNode(new int[]{2, 2, 2});
        InputNode b = new InputNode(new int[]{2, 2, 2});
        Node output = new AdditionNode(a, b);

        Tensor aI = new Tensor(new float[]{
                1, 2,
                5, 10,

                5, 2,
                0, 0
        }, new int[]{2, 2, 2});
        Tensor bI = new Tensor(new float[]{
                1, 2,
                5, 10,

                5, 5,
                1, 0
        }, new int[]{2, 2, 2});
        Eval e = new Eval()
                .addInput(a, aI)
                .addInput(b, bI);
        Assertions.assertTrue(e.evaluate(output).equals(new Tensor(new float[]{
                2, 4,
                10, 20,

                10, 7,
                1, 0
        }, new int[]{2, 2, 2})));
    }

    @Test
    void evaluate4() {
        InputNode a = new InputNode(new int[]{2, 2, 2});
        InputNode b = new InputNode(new int[]{1});
        Node output = new AdditionNode(a, b);

        Tensor aI = new Tensor(new float[]{
                1, 2,
                5, 10,

                5, 2,
                0, 0
        }, new int[]{2, 2, 2});
        Tensor bI = new Tensor(new float[]{
                5
        }, new int[]{1});
        Eval e = new Eval()
                .addInput(a, aI)
                .addInput(b, bI);
        Assertions.assertTrue(e.evaluate(output).equals(new Tensor(new float[]{
                6, 7,
                10, 15,

                10, 7,
                5, 5
        }, new int[]{2, 2, 2})));
    }

    @Test
    void evaluate5() {
        InputNode a = new InputNode(new int[]{1});
        InputNode b = new InputNode(new int[]{2, 2, 2});
        Node output = new AdditionNode(a, b);

        Tensor aI = new Tensor(new float[]{
                5
        }, new int[]{1});
        Tensor bI = new Tensor(new float[]{
                1, 2,
                5, 10,

                5, 2,
                0, 0
        }, new int[]{2, 2, 2});
        Eval e = new Eval()
                .addInput(a, aI)
                .addInput(b, bI);
        Assertions.assertTrue(e.evaluate(output).equals(new Tensor(new float[]{
                6, 7,
                10, 15,

                10, 7,
                5, 5
        }, new int[]{2, 2, 2})));
    }

}