import * as tf from "@tensorflow/tfjs";
import { useRef, useState } from "react";
import Webcam from "react-webcam";
import styled from "styled-components";

import useAnimationFrame from "./hooks/useAnimationFrame";
import useAsync from "./hooks/useAsync";

const OuterDiv = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
`;

const InnerDiv = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    width: 100%;
`;

const FlexWebcam = styled(Webcam)`
    flex-grow: 1;
`;

const Canvas = styled.canvas`
    flex-grow: 1;
`;

const videoConstraints = {
    width: 256,
    height: 256
};

const screenShotDimensions = {
    width: 256,
    height: 256
};

const colorMap = tf.tensor2d([[0, 0, 0], [1, 1, 1]]);

function decodeSegmentationMasks(
    mask: tf.Tensor<tf.Rank.R2>,
    colorMap: tf.Tensor<tf.Rank.R2>,
    classCount: number
): tf.Tensor<tf.Rank.R2> {
    let red = tf.zerosLike(mask);
    let green = tf.zerosLike(mask);
    let blue = tf.zerosLike(mask);

    for (let i = 0; i < classCount; i++) {
        const color = colorMap.gather(i);
        const maskEqualToClass = tf.equal(mask, i);

        const colorRed = color.gather(0);
        const colorGreen = color.gather(1);
        const colorBlue = color.gather(2);

        const multipliedRed = tf.mul(maskEqualToClass, colorRed);
        const multipliedGreen = tf.mul(maskEqualToClass, colorGreen);
        const multipliedBlue = tf.mul(maskEqualToClass, colorBlue);

        const newRed = tf.add(red, multipliedRed);
        const newGreen = tf.add(green, multipliedGreen);
        const newBlue = tf.add(blue, multipliedBlue);

        red.dispose();
        green.dispose();
        blue.dispose();

        red = newRed as tf.Tensor<tf.Rank.R2>;
        green = newGreen as tf.Tensor<tf.Rank.R2>;
        blue = newBlue as tf.Tensor<tf.Rank.R2>;

        multipliedRed.dispose();
        multipliedGreen.dispose();
        multipliedBlue.dispose();

        colorRed.dispose();
        colorGreen.dispose();
        colorBlue.dispose();

        maskEqualToClass.dispose();
        color.dispose();
    }

    const stack = tf.stack([red, green, blue], 2) as tf.Tensor<tf.Rank.R2>;
    blue.dispose();
    green.dispose();
    red.dispose();
    return stack;
}

function App(): JSX.Element {
    const [ model, setModel ] = useState<tf.LayersModel | null>(null);
    const webcamRef = useRef<Webcam>(null);
    const resultCanvasRef = useRef<HTMLCanvasElement>(null);

    useAsync(async () => {
        const model = await tf.loadLayersModel("tfjs-model/model.json");
        setModel(model);
    }, []);

    useAnimationFrame(() => {
        if (!model) return;
        if (!webcamRef.current) return;

        const imageSrc = webcamRef.current.getScreenshot(screenShotDimensions);
        if (!imageSrc) return;

        const image = new Image();
        image.src = imageSrc;
        image.onload = async (): Promise<void> => {
            const tensor = tf.browser.fromPixels(image);
            const tensorExpanded = tf.expandDims(tensor, 0);
            const prediction = model.predict(tensorExpanded) as tf.Tensor<tf.Rank.R4>;
            const predictionSqueezed = tf.squeeze(prediction);
            const predictionArgMax = tf.argMax(predictionSqueezed, 2) as tf.Tensor<tf.Rank.R2>;
            const predictionColorMap = decodeSegmentationMasks(predictionArgMax, colorMap, 2);

            const resultCanvas = resultCanvasRef.current;
            if (!resultCanvas) return;

            await tf.browser.toPixels(predictionColorMap, resultCanvas);

            predictionColorMap.dispose();
            predictionArgMax.dispose();
            predictionSqueezed.dispose();
            prediction.dispose();
            tensorExpanded.dispose();
            tensor.dispose();

            URL.revokeObjectURL(image.src);
        };
    });

    return (
        <OuterDiv>
            <InnerDiv>
                <FlexWebcam 
                    videoConstraints={videoConstraints}
                    ref={webcamRef}
                />
                <Canvas 
                    width={screenShotDimensions.width}
                    height={screenShotDimensions.height}
                    ref={resultCanvasRef}
                />
            </InnerDiv>
        </OuterDiv>
    );
}

export default App;
