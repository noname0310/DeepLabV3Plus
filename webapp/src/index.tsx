import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { loadDeepLabV3Plus } from "./deeplabv3plus";

const model = loadDeepLabV3Plus();
model;

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
