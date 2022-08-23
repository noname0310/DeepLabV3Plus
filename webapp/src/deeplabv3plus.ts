import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";

export class TFOpLambda extends tf.layers.Layer {
    public static className = "TFOpLambda";

    public constructor(config?: LayerArgs) {
        super(config);
    }

    public override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return inputShape;
    }

    public override call(_inputs: tf.Tensor | tf.Tensor[], _kwargs: any): tf.Tensor | tf.Tensor[] {
        return tf.tidy(() => {
            return tf.scalar(1.0);
        });
    }
}

tf.serialization.registerClass(TFOpLambda);

export function loadDeepLabV3Plus(): Promise<tf.GraphModel> {
    return tf.loadGraphModel("tfjs-model/model.json");
}
