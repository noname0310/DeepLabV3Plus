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
    width: 100vw;
`;

const videoConstraints = {
    width: 256,
    height: 256
};

const screenShotDimensions = {
    width: 256,
    height: 256
};

const colorMap = tf.tensor2d([[0, 0, 0], [0, 0, 255]]);

function decodeSegmentationMasks(
    mask: tf.Tensor<tf.Rank.R2>,
    colorMap: tf.Tensor<tf.Rank.R2>,
    classCount: number
): tf.Tensor<tf.Rank.R2> {
    const redZero = tf.zerosLike(mask);
    const red = redZero.asType("int32");
    const greenZero = tf.zerosLike(mask);
    const green = greenZero.asType("int32");
    const blueZero = tf.zerosLike(mask);
    const blue = blueZero.asType("int32");

    for (let i = 0; i < classCount; i++) {
        const color = colorMap.gather(i);
        const maskEqualToClass = tf.equal(mask, i);

        const colorRed = color.gather(0);
        const colorGreen = color.gather(1);
        const colorBlue = color.gather(2);

        const multipliedRed = tf.mul(maskEqualToClass, colorRed);
        const multipliedGreen = tf.mul(maskEqualToClass, colorGreen);
        const multipliedBlue = tf.mul(maskEqualToClass, colorBlue);

        red.add(multipliedRed);
        green.add(multipliedGreen);
        blue.add(multipliedBlue);

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
    blueZero.dispose();
    green.dispose();
    greenZero.dispose();
    red.dispose();
    redZero.dispose();
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
            <Webcam 
                videoConstraints={videoConstraints}
                ref={webcamRef}
                width={"80%"}
            />
            <canvas 
                width={screenShotDimensions.width}
                height={screenShotDimensions.height}
                ref={resultCanvasRef}
            />
        </OuterDiv>
    );
}

export default App;
