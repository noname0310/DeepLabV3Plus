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

function App(): JSX.Element {
    const [ model, setModel ] = useState<tf.LayersModel | null>(null);
    const webcamRef = useRef<Webcam>(null);

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
            // print dimensions of tensor
            console.log(tensor.shape);

            // const prediction = model.predict(tensor);
            
            // console.log(prediction);
        };
    });

    return (
        <OuterDiv>
            <Webcam 
                videoConstraints={videoConstraints}
                ref={webcamRef}
                width={"80%"}
            />
        </OuterDiv>
    );
}

export default App;
