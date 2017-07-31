package com.github.cg4j;

import com.github.cg4j.exception.IllegalShapeException;
import com.github.cg4j.nodes.InputNode;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class EvalTest {

    @Test
    void addInput() {
        InputNode a = new InputNode(new int[]{4, -1, 3});
        Tensor tensor = new Tensor(new float[4 * 3 * 3], new int[]{4, 3, 3});
        Eval e = new Eval().addInput(a, tensor);
        Assertions.assertEquals(tensor, e.evaluate(a));

        try {
            e.addInput(a, new Tensor(new float[4 * 3], new int[]{4, 3}));
            Assertions.assertTrue(false);
        } catch (IllegalShapeException ise) {
            Assertions.assertTrue(true);
        }

        try {
            e.addInput(a, new Tensor(new float[4 * 3 * 2], new int[]{4, 3, 2}));
            Assertions.assertTrue(false);
        } catch (IllegalShapeException ise) {
            Assertions.assertTrue(true);
        }
    }

}