import { DependencyList, useEffect, useState } from "react";

import useMountedState from "./useMountedState";

function useAnimationFrame(callback: () => void, deps: DependencyList = []): void {
    const [_, setRequestAnimationFrameId] = useState(0);
    const isMounted = useMountedState();

    useEffect((): (void | (() => void)) => {
        const id = requestAnimationFrame(() => {
            isMounted() && setRequestAnimationFrameId(id);
            callback();
        });

        return () => cancelAnimationFrame(id);
    }, [callback, isMounted, ...deps]);
}

export default useAnimationFrame;
