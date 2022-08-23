import { DependencyList, useEffect } from "react";

import { FunctionReturningPromise } from "./misc/types";
import useAsyncFn, { StateFromFunctionReturningPromise } from "./useAsyncFn";

export default function useAsync<T extends FunctionReturningPromise>(
    fn: T,
    deps: DependencyList = []
): StateFromFunctionReturningPromise<T> {
    const [state, callback] = useAsyncFn(fn, deps, {
        loading: true
    });

    useEffect(() => {
        callback();
    }, [callback]);

    return state;
}
