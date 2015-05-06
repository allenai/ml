package com.allenai.ml.sequences;

import lombok.Data;
import lombok.Value;


/**
 * A transition represents `(from, to)` transition between states. We store the `selfIndex` of the from and to state
 * as well as its index relative to other transitions. This is meant to only be used internal to this package.
 *
 *  __Note__: Default access-level is intentional
 * @See StateSpace
 */
@Data(staticConstructor = "of")
class Transition {
    public final int fromState;
    public final int toState;
    public final int selfIndex;
}
