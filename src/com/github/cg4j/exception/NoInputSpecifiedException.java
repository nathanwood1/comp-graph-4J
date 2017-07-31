package com.github.cg4j.exception;

/**
 * This exception represents inputs being required but not specified.
 * An example would be:
 * <pre>
 * {@code
 *
 * InputNode a = new InputNode(new int[]{1});
 * Node out = new ReLUNode(a);
 *
 * Eval e = new Eval();
 * e.evaluate(out);
 * }
 * </pre>
 * @see com.github.cg4j.nodes.InputNode
 * @since 1.0
 * @author nathanwood1
 */
public class NoInputSpecifiedException extends IllegalArgumentException {

    public NoInputSpecifiedException(String str) {
        super(str);
    }
}
